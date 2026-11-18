"""Run the isolated histogram status-mask trial against one frozen optimized provider."""

import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import typer

TRIAL = Path(__file__).resolve().parent
ROOT = TRIAL.parents[2]
REFERENCE = ROOT / "results/reference"
sys.path.insert(0, str(ROOT / "tools/reference"))
from benchmark import CPU, check_quality, run_leg
sys.path.insert(0, str(ROOT))
from tools.benchmark_campaign import clean_environment, identity

SCOPES = {"small": ("small", 1000), "large": ("large", 4000), "both": ("combined", 1000)}
VARIANTS = ("control", "skip-decorations")
SOURCES = (
    "columnar/jets.py", "columnar/inputs.py", "jet_transport.py",
    "tools/compare_columnar.py", "config/jet-states.json", "config/jet-histogram.json",
)
PARTITION = ROOT / "results/export/eta-jes-partition-001"
BUILD = ROOT / "results/build/core-precision-002"
SDK = BUILD / "venv/lib/python3.14/site-packages/correctionlib"
PYTHON = BUILD / "venv/bin/python"
GCC = Path("/cvmfs/sft.cern.ch/lcg/releases/gcc/15.2.0-35657/x86_64-el9/lib64")


def read_json(path):
    return json.loads(path.read_text())


def receipt(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def frozen_sources(output):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    destinations = {}
    for arm in VARIANTS:
        root = output / "sources" / arm
        for relative in SOURCES:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            content = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
            path.write_bytes(content)
        destinations[arm] = root
    subprocess.run(["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(TRIAL / "integration.diff")],
                   cwd=destinations["skip-decorations"], stdout=subprocess.PIPE, check=True)
    digests = {arm: {name: identity(root / name) for name in SOURCES} for arm, root in destinations.items()}
    differences = [name for name in SOURCES if digests["control"][name] != digests["skip-decorations"][name]]
    if differences != ["columnar/jets.py"]:
        raise ValueError(f"Unexpected source difference between trial snapshots: {differences}")
    return destinations, {"git_source": commit, "snapshots": digests, "delta": identity(TRIAL / "integration.diff")}


def source_assets(provider):
    core = SDK / "lib/libcorrectionlib.so"
    source = read_json(ROOT / "config/jets-input.json")
    original = Path(source["path"]).resolve(strict=True)
    if identity(original) != {key: source[key] for key in ("sha256", "bytes")}:
        raise ValueError("Original file differs from the stock observer's frozen sample")
    native = provider.parent
    choices = set((native / "build-choices.txt").read_text().splitlines())
    baseline = {"rng-cache-large", "fast-variable-flow", "unsigned-sign-pairs"}
    if choices not in (baseline, baseline | {"rng-prefix"}) or (native / "exit-code").read_text().strip() != "0":
        raise ValueError(f"Wrong optimized provider options: {sorted(choices)}")
    if (native / "rng-cache-slots.txt").read_text().strip() != "262144":
        raise ValueError("The trial requires the selected 4 MiB Gaussian cache")
    recorded = {}
    for line in (native / "native.sha256").read_text().splitlines():
        sha, file = line.split(maxsplit=1)
        recorded[(native / file).resolve()] = sha
    for path in (provider, core):
        if identity(path)["sha256"] != recorded.get(path.resolve()):
            raise ValueError(f"Native source does not match its recorded artifact: {path}")

    manifest = read_json(PARTITION / "manifest.json")
    index = read_json(PARTITION / "payload-index.json")
    provenance = Path(manifest["provenance"]["path"]).resolve(strict=True)
    if identity(provenance)["sha256"] != manifest["provenance"]["sha256"]:
        raise ValueError("Packed payload provenance changed")
    scopes = {}
    for scope, (reference, chunk) in SCOPES.items():
        relative = manifest["native_load_paths"][reference]
        selected = (PARTITION / relative).resolve(strict=True)
        expected = index["canonical"] if scope == "both" else index["scopes"][scope]
        if identity(selected) != {key: expected[key] for key in ("bytes", "sha256")}:
            raise ValueError(f"Unexpected packed {scope} CorrectionSet")
        archive = REFERENCE / f"full-file-pilot-{reference}-002/01-full/histograms.npz"
        with np.load(archive, allow_pickle=False) as authority:
            if authority["events"].item() != 20000:
                raise ValueError(f"Stock endpoint does not cover the full sample: {archive}")
        previous = read_json(ROOT / f"results/packaging/scope-check-{scope}-001/comparison.json")
        branches = set(previous["branch_digests"].values())
        if len(branches) != 1 or not previous["all_histogram_fields_exact"]:
            raise ValueError(f"Missing independent physical branch archive for {scope}")
        scopes[scope] = {
            "payload": str(selected), "payload_identity": identity(selected),
            "stock_histogram": str(archive), "stock_histogram_identity": identity(archive),
            "branch_reference": {"path": str(ROOT / f"results/packaging/scope-check-{scope}-001/comparison.json"),
                                 "physical_sha256": branches.pop()},
            "physical_chunk": chunk,
        }
    observer = REFERENCE / "full-clean-001"
    checked = read_json(observer / "verification.json")
    if checked["events"] != 20000 or checked["source_file_uuid"] != source["tfile_uuid"]:
        raise ValueError("Independent observer does not cover the exact frozen original sample")
    asset_receipt = {
        "stock_observer": {"path": str(observer / "reference.root"),
                           "sha256": (observer / "reference.sha256").read_text().split()[0],
                           "verification": identity(observer / "verification.json")},
        "source": source, "resolved_original": str(original), "core": {"path": str(core), **identity(core)},
        "provider": {"path": str(provider), **identity(provider), "choices": sorted(choices)},
        "partition": str(PARTITION), "provenance": {"path": str(provenance), **identity(provenance)},
        "scopes": scopes,
    }
    return original, provenance, scopes, asset_receipt


class CountWhere:
    def __init__(self):
        self.calls = 0

    def __getattr__(self, key):
        return getattr(np, key)

    def where(self, *args):
        self.calls += 1
        return np.where(*args)


def status_function(path, calls):
    top = ast.parse(path.read_text())
    definition, = (node for node in top.body if isinstance(node, ast.FunctionDef) and node.name == "status_bits")
    namespace = {"np": calls, "VALID": np.uint32(0xFFFFFFFF), "OUT_OF_VALIDITY": np.uint32(0xFFFFFFFE)}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["status_bits"]


def synthetic_status_oracle(output, sources):
    counters = {name: CountWhere() for name in sources}
    checks = {name: status_function(path / "columnar/jets.py", counters[name]) for name, path in sources.items()}
    cases = [
        (np.array([2.0, 1.0, 0., 255., np.nan]), np.array([1, 1, 0, 0, 0]),
         np.array([0xFFFFFFFF, 0xFFFFFFFE, 0, 0, 0], dtype=np.uint32)),
        (np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.uint32)),
    ]
    errors = [np.array([status], dtype=np.float64) for status in (0., -1., 3., np.inf, np.nan)]
    for name in ("sr NOSYS uncertainty", "sr JET_NNJvtEfficiency__1up uncertainty",
                 "lr JET_JER_EffectiveNP_1__1up uncertainty", "lr JMR FF state"):
        control, candidate = (checks[arm] for arm in VARIANTS)
        for status, present, want in cases:
            np.testing.assert_array_equal(control(status, present, name), want)
            np.testing.assert_array_equal(candidate(status, present, name, materialize=True), want)
            before = counters["skip-decorations"].calls
            if candidate(status, present, name, materialize=False) is not None or before != counters["skip-decorations"].calls:
                raise ValueError(f"The histogram path materialized diagnostic flags for {name}")
        for status in errors:
            for evaluate in (lambda: control(status, np.array([1]), name),
                             lambda: candidate(status, np.array([1]), name, materialize=True),
                             lambda: candidate(status, np.array([1]), name, materialize=False)):
                try:
                    evaluate()
                except ValueError as error:
                    if str(error) != f"{name} returned a release correction error":
                        raise
                else:
                    raise ValueError(f"Accepted a present invalid status in {name}")
    receipt(output / "synthetic-status.json", {
        "exact_diagnostic_bits": True, "histogram_materializes_no_bits": True, "exact_errors": True,
        "scope": "Native numeric storage layout, including absent rows, empty rows and present invalid statuses",
    })


def common_args(scope, packed, provider, provenance):
    args = ["--provider", str(provider), "--payload", packed["payload"], "--provenance", str(provenance),
            "--scope", scope, "--mode", "full", "--reuse-nominal-nnjvt-update", "--matched-read"]
    if scope != "large":
        args.append("--reuse-uncertainty-sign-pairs")
    return args


def check_oracle(output, sources, original, provider, provenance, scopes, env):
    saved = REFERENCE / "full-clean-001/reference.root"
    for scope, packed in scopes.items():
        for arm in VARIANTS:
            job = output / "oracle" / scope / arm
            job.mkdir(parents=True)
            args = [flag for flag in common_args(scope, packed, provider, provenance) if flag != "--matched-read"]
            run_leg([str(PYTHON), str(sources[arm] / "tools/compare_columnar.py"), str(original), str(saved),
                     str(job / "comparison.json"), "--events", "32", "--chunk", "19", *args],
                    env, job, "observer")
        a, b = (read_json(output / "oracle" / scope / arm / "comparison.json") for arm in VARIANTS)
        if (not a["passed"] or not b["passed"] or a["source_file_uuid"] != b["source_file_uuid"]
                or a["entries_compared"] != b["entries_compared"] or b["entries_compared"] != 32
                or a["states"] != b["states"] or a["checks"] != b["checks"] or a["runtime"] != b["runtime"]):
            raise ValueError(f"Diagnostic fields or native runtime changed in the {scope} oracle")


def check_output(job, original_uuid, provider, spec, leg):
    wanted = {
        provider.name: provider.resolve(),
        "libcorrectionlib.so": (SDK / "lib/libcorrectionlib.so").resolve(),
    }
    for name, required in wanted.items():
        observed = {Path(path).resolve() for path in leg["libraries"] if Path(path).name == name}
        if observed != {required}:
            raise ValueError(f"Unexpected native map for {name} in {job}: {observed}")
    log = (job / "direct.log").read_text()
    branches = re.findall(r"(?m)^Input branch SHA-256: ([0-9a-f]{64})$", log)
    if (not leg["source_mapping_observed"] or log.count(f"({original_uuid})") != 1
            or branches != [spec["branch_reference"]["physical_sha256"]]
            or log.count("20000 unique events, 20000 event visits across 1 passes") != 1):
        raise ValueError(f"Timed source identity, map, branch list or visit count changed in {job}")
    with np.load(job / "histograms.npz", allow_pickle=False) as actual, np.load(
        spec["stock_histogram"], allow_pickle=False
    ) as stock:
        if set(actual.files) != set(stock.files):
            raise ValueError(f"Stock and candidate histogram fields differ in {job}")
        for field in stock.files:
            np.testing.assert_array_equal(actual[field], stock[field], err_msg=field)
    return {"branch_sha256": branches[0], "mapped_source": True,
            "histogram": identity(job / "histograms.npz"), "resource_receipt": identity(job / "direct.json")}


def time_pairs(output, sources, original, provider, provenance, scopes, assets, env):
    observations = {scope: {arm: [] for arm in VARIANTS} for scope in scopes}
    cases = list(scopes)
    for round_index in range(4):
        shift = round_index % len(cases)
        for order, scope in enumerate(cases[shift:] + cases[:shift]):
            packed = scopes[scope]
            visit = VARIANTS if (round_index + order) % 2 == 0 else VARIANTS[::-1]
            for arm in visit:
                job = output / "timing" / f"{round_index:02d}-{scope}-{arm}"
                job.mkdir(parents=True)
                command = [str(PYTHON), str(sources[arm] / "columnar/jets.py"), str(original),
                           str(job / "histograms.npz"), "--events", "20000", "--chunk", str(packed["physical_chunk"]),
                           *common_args(scope, packed, provider, provenance)]
                leg = run_leg(command, env, job, "direct", event_source=original)
                verified = check_output(job, assets["source"]["tfile_uuid"], provider, packed, leg)
                observations[scope][arm].append(
                    {"round": round_index, "wall_s": leg["wall_s"], "cpu_s": leg["cpu_user_s"] + leg["cpu_sys_s"],
                     "peak_RSS_kB": leg["peak_rss_kb"], **verified}
                )
    results = {}
    for scope, arms in observations.items():
        left, right = (arms[arm] for arm in VARIANTS)
        quality = {arm: check_quality(samples) for arm, samples in arms.items()}
        results[scope] = {
            "gate_all_arms": all(rule["pass"] for rule in quality.values()),
            "quality": quality,
            "control_over_candidate_mean": quality["control"]["mean_s"] / quality["skip-decorations"]["mean_s"],
            "retained_same_round_ratios": [a["wall_s"] / b["wall_s"] for a, b in zip(left[1:], right[1:], strict=True)],
            "observations": arms,
        }
    receipt(output / "summary.json", {
        "results": results, "resource_review_required": True, "all_histograms_exact": True,
        "timing_boundary": "Checkout CLI and prebuilt scoped packed documents. Real release file and saved histogram.",
        "production_compact_reconstruction_in_timed_interval": False,
    })


def main(provider: Path, output: Path, oracle_only: bool = False):
    """Compare the selected provider with and without unused Python status decorations."""
    if os.sched_getaffinity(0) != {CPU} or len(list(Path("/proc/self/task").iterdir())) != 1:
        raise ValueError("Run only on the reserved CPU3 with one execution thread")
    provider = provider.resolve(strict=True)
    output = output.resolve()
    original, provenance, scopes, assets = source_assets(provider)
    output.mkdir()
    sources, snapshot = frozen_sources(output)
    receipt(output / "inputs.json", {"assets": assets, "source": snapshot, "time": not oracle_only})
    env = clean_environment()
    env["LD_LIBRARY_PATH"] = f"{SDK / 'lib'}:{GCC}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    synthetic_status_oracle(output, sources)
    check_oracle(output, sources, original, provider, provenance, scopes, env)
    if not oracle_only:
        time_pairs(output, sources, original, provider, provenance, scopes, assets, env)


if __name__ == "__main__":
    typer.run(main)
