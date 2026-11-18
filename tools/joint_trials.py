"""Verify the joint-build fixtures and prepare independent single-mechanism C pairs."""

from hashlib import file_digest
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Annotated

import typer

APP = typer.Typer(add_completion=False)
ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "results/build"
EXPORT = ROOT / "results/export"
REFERENCE = ROOT / "results/reference"
COMPILER_SETUP = "/cvmfs/sft.cern.ch/lcg/releases/gcc/15.2.0-35657/x86_64-el9/setup.sh"
SCOPES = ("small", "large", "both")
FAMILIES = ("rng-size", "variable-flow", "jmr-match", "generic-core", "packed-eta-jes", "packed-scope")
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "ROOT_MAX_THREADS", "RAYON_NUM_THREADS")


def emit(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def json_file(path):
    return json.loads(path.read_text())


def checksums(path):
    values = {}
    for line in path.read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        filename = Path(filename.lstrip("*"))
        if not filename.is_absolute():
            filename = path.parent / filename
        if filename in values or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid checksum record in {path}: {line}")
        values[filename] = digest
    return values


class Build:
    def __init__(self, label):
        if not re.fullmatch(r"[a-z0-9-]+", label):
            raise ValueError("Expected the exact joint build suffix")
        self.label = label
        self.work = BUILD / f"joint-trials-{label}"
        self.status = {}
        for line in (self.work / "status.txt").read_text().splitlines():
            status, name = line.split()
            if name in self.status:
                raise ValueError(f"Duplicate build status for {name}")
            self.status[name] = int(status)
        self.regular = self.runtime("core-precision-002")
        self.generic = self.runtime(f"core-plugin-batch-{label}")

    def require(self, *names):
        missing = {name: self.status.get(name) for name in names if self.status.get(name) != 0}
        if missing:
            raise ValueError(f"Missing or failed joint build items: {missing}")

    def runtime(self, core):
        directory = BUILD / core
        sdk = directory / "venv/lib/python3.14/site-packages/correctionlib"
        return {"python": str(directory / "venv/bin/python"), "sdk": str(sdk)}

    def provider(self, name):
        self.require(name)
        result = BUILD / f"joint-{name}-{self.label}"
        sdk = self.generic["sdk"] if name == "generic" else self.regular["sdk"]
        provider, core = result / "libjets.so", Path(sdk) / "lib/libcorrectionlib.so"
        hashes = checksums(result / "native.sha256")
        for expected in (provider, core):
            if hashes.get(expected) != sha(expected):
                raise ValueError(f"Wrong native build bytes for {name}: {expected}")
        expected_flags = {
            "baseline": {"rng-cache"}, "bigger": {"rng-cache-large"},
            "flow": {"rng-cache", "fast-variable-flow"}, "match": {"rng-cache", "large-jmr-match-cache"},
            "generic": {"rng-cache"},
        }[name]
        if set((result / "build-choices.txt").read_text().splitlines()) != expected_flags:
            raise ValueError(f"Unexpected native choices for {name}")
        return provider

    def check_sdk_pair(self):
        def package_lock(runtime):
            path = Path(runtime["python"]).parents[2] / "packages.txt"
            return {line for line in path.read_text().splitlines() if not line.startswith(("correctionlib ", "correctionlib=="))}

        def python_version(runtime):
            path = Path(runtime["python"]).parents[2] / "toolchain.txt"
            return next(line for line in path.read_text().splitlines() if line.startswith("Python "))

        if (package_lock(self.regular) != package_lock(self.generic)
                or python_version(self.regular) != python_version(self.generic)):
            raise ValueError("Generic-core comparison requires identical non-CorrectionLib packages and Python versions")


def clean_env():
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD"):
        env.pop(key, None)
    env.update(dict.fromkeys(THREAD_ENV, "1"))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def resolve_environments(directory):
    base = clean_env()
    gcc = subprocess.check_output(
        ["bash", "-c", 'source "$1" >&2 || exit; dirname "$(g++ -print-file-name=libstdc++.so)"',
         "setup-gcc", COMPILER_SETUP], cwd=directory, env=base, text=True).strip()
    compiler = {**base, "LD_LIBRARY_PATH": str(Path(gcc).resolve(strict=True))}
    with (directory / "release-setup.log").open("x") as log:
        stock = subprocess.check_output(
            ["bash", "-c", 'source "$1" >&2 || exit; env -0',
             "setup-release", str(ROOT / "tools/reference/setup.sh")],
            cwd=directory, env=base, stderr=log,
        )
    release = dict(item.split("=", 1) for item in os.fsdecode(stock).rstrip("\0").split("\0"))
    return compiler, release


def fixture_environment(runtime, compiler):
    return {**compiler, "LD_LIBRARY_PATH": f"{runtime['sdk']}/lib:{compiler['LD_LIBRARY_PATH']}"}


def verify_links(binary, libraries, env, directory):
    with (directory / "ldd.txt").open("x") as stream:
        proc = subprocess.run(["ldd", str(binary)], env=env, text=True, stdout=stream, stderr=subprocess.STDOUT)
    lines = (directory / "ldd.txt").read_text().splitlines()
    if proc.returncode or any("not found" in line for line in lines):
        raise ValueError(f"Unresolved dynamic fixture dependencies for {binary.name}")
    linked = {}
    for line in lines:
        row = re.match(r"\s*(\S+)(?:\s*=>\s*(\S+))?", line)
        if row:
            target = Path(row[2] or row[1])
            if target.is_absolute():
                linked.setdefault(Path(row[1]).name, set()).add(target.resolve())
    for name, expected in libraries.items():
        paths = linked.get(name, set())
        if paths != {expected.resolve()}:
            raise ValueError(f"{binary.name}: expected {name} from {expected}, got {paths}")


@APP.command()
def fixtures(label: str, result: Path):
    """Run the compiled focused fixtures on the reserved CPU, retaining all case outcomes."""
    if os.sched_getaffinity(0) != {3} or len(list(Path("/proc/self/task").iterdir())) != 1:
        raise ValueError("Fixtures require the reserved CPU3 and one controller thread")
    build = Build(label)
    build.require("focused-fixtures", "fixture-original", "fixture-generic", "match")
    match = build.provider("match")
    result = result.resolve()
    result.mkdir()
    compiler, release = resolve_environments(result)
    fixture = build.work / "fixtures"
    codec = r"^Exact packed binary32 transport and decimal histogram equality passed$"
    oracle = r"^exact release comparisons: [1-9][0-9]*$"
    cases = {
        "flow-plain": (fixture / "flow-plain", compiler, {}, (), None),
        "flow-flow": (fixture / "flow-flow", compiler, {}, (), None),
        "oracle-plain": (fixture / "oracle-plain", release, {}, (), oracle),
        "oracle-flow": (fixture / "oracle-flow", release, {}, (), oracle),
        "packed-portable": (fixture / "packed-portable", compiler, {}, (), codec),
        "packed-stock": (fixture / "packed-stock", release, {}, (), codec),
        "rng-large": (fixture / "rng-large", release, {}, (), r"^bitwise ROOT checks [1-9][0-9]*$"),
        "jmr-match": (fixture / "jmr-match", fixture_environment(build.regular, compiler),
                      {"libjets.so": match, "libcorrectionlib.so": Path(build.regular["sdk"]) / "lib/libcorrectionlib.so"},
                      (str(match), str(EXPORT / "portable-001/jets.json")),
                      r"^252 exact six-value comparisons across 36 FF states, plus lifetime and match-order edges$"),
    }
    for variant, runtime in (("original", build.regular), ("generic", build.generic)):
        folder = build.work / f"{variant}-sdk-fixture"
        cases[f"core-{variant}"] = (
            folder / "core-plugin-batch-fixture", fixture_environment(runtime, compiler),
            {"libcorrectionlib.so": Path(runtime["sdk"]) / "lib/libcorrectionlib.so"}, (),
            r"^generic plugin batch fixture passed$",
        )
    registry = checksums(fixture / "binaries.sha256")
    receipts = {}
    for label, (binary, env, links, arguments, marker) in cases.items():
        folder = result / label
        folder.mkdir()
        log = folder / "run.log"
        command = [str(binary), *arguments]
        record = {"command": command, "passed": False, "exit": None}
        try:
            expected = (checksums(binary.parent / "fixture.sha256") if label.startswith("core-") else registry)
            for path in {binary, *links.values()} & expected.keys():
                if sha(path) != expected[path]:
                    raise ValueError(f"Fixture build checksum changed: {path}")
            if binary not in expected:
                raise ValueError(f"Fixture has no recorded ODR/build identity: {binary}")
            if label.startswith("core-") and not set(links.values()) <= expected.keys():
                raise ValueError(f"Fixture has no recorded SDK identity: {binary}")
            record["binary_sha256"] = expected[binary]
            if label == "jmr-match":
                payload = Path(arguments[1])
                record["payload"] = {"path": str(payload), "sha256": sha(payload),
                                     "provenance_sha256": sha(payload.with_name("provenance.json"))}
            verify_links(binary, links, env, folder)
            with log.open("x") as stdout, (folder / "stderr.log").open("x") as stderr:
                completed = subprocess.run(command, cwd=folder, env=env, stdout=stdout, stderr=stderr)
            record["exit"] = completed.returncode
            if completed.returncode:
                raise ValueError(f"Fixture returned {completed.returncode}")
            output = log.read_text()
            if marker and not re.search(marker, output, re.MULTILINE):
                raise ValueError("Fixture completion marker was missing")
            if label.startswith("flow-"):
                blocks = output.splitlines()
                values = [line for line in blocks if not line.startswith("case ")]
                if len(blocks) - len(values) != 21 or len(values) != 74556 or any(
                        not re.fullmatch(r"(?:[0-9a-f]{1,16}|range-error)", line) for line in values):
                    raise ValueError("Incomplete exact flow fixture")
            if label == "rng-large" and not re.search(r"^cache_slots 262144$", output, re.MULTILINE):
                raise ValueError("Large resolution cache size was not active")
            record.update(log_sha256=sha(log), passed=True)
        except (OSError, ValueError) as error:
            record["error"] = str(error)
        emit(folder / "receipt.json", record)
        receipts[label] = record
    pairs = {}
    for name, a, c in (
        ("variable_flow_bits", "flow-plain", "flow-flow"),
        ("stock_oracle_coverage", "oracle-plain", "oracle-flow"),
        ("scalar_sdk_fixture", "core-original", "core-generic"),
    ):
        pairs[name] = receipts[a]["passed"] and receipts[c]["passed"] and (
            receipts[a]["log_sha256"] == receipts[c]["log_sha256"])
    passed = all(row["passed"] for row in receipts.values()) and all(pairs.values())
    emit(result / "summary.json", {"passed": passed, "fixtures": receipts, "differential_checks": pairs})
    if not passed:
        raise typer.Exit(1)


def verify_payload(path, identity):
    if path.stat().st_size != identity["bytes"] or sha(path) != identity["sha256"]:
        raise ValueError(f"Payload bytes disagree with the frozen identity at {path}")


@APP.command()
def plan(
    label: str, result: Path,
    rounds: Annotated[int, typer.Option(min=1)] = 1,
    screen_events: Annotated[int, typer.Option(min=1, max=20000)] = 32,
    nnjvt_reuse: bool = False,
):
    """Freeze one independent original-file pair per mechanism before screening."""
    if result.exists():
        raise FileExistsError(result)
    build = Build(label)
    build.require("pack", "partition", "scope-index", "generic-core")
    providers = {name: build.provider(name) for name in ("baseline", "bigger", "flow", "match", "generic")}
    build.check_sdk_pair()
    source_list = {name: checksums(path.parent / "source-inputs.sha256") for name, path in providers.items()}
    if any(inputs != source_list["baseline"] for inputs in source_list.values()):
        raise ValueError("Joint providers were built from different source snapshots")

    original = EXPORT / "portable-001/jets.json"
    packed_dir = EXPORT / f"eta-jes-packed-{label}"
    packed = packed_dir / "jets.json"
    pack_proof = json_file(packed_dir / "packing.json")
    scoped_dir = EXPORT / f"eta-jes-partition-{label}"
    partition = json_file(scoped_dir / "manifest.json")
    index = json_file(scoped_dir / "payload-index.json")
    verify_payload(original, pack_proof["canonical"])
    verify_payload(packed, pack_proof["packed"])
    verify_payload(packed, index["canonical"])
    verify_payload(packed, partition["canonical"])
    if sha(packed_dir / "provenance.json") != partition["provenance"]["sha256"]:
        raise ValueError("Packed payload provenance differs from the partition input")
    if not partition["raw_node_identity_verified"]:
        raise ValueError("Missing byte-level proof of the packed partition")

    projected = {}
    for scope in ("small", "large"):
        path = scoped_dir / partition["native_load_paths"][scope]
        verify_payload(path, index["scopes"][scope])
        if sha(path.with_name("provenance.json")) != partition["provenance"]["sha256"]:
            raise ValueError(f"Unexpected {scope} payload provenance")
        projected[scope] = path

    config_file = ROOT / "config/jets-input.json"
    source = json_file(config_file)
    input_path = Path(source["path"])
    if input_path.stat().st_size != source["bytes"]:
        raise ValueError("Original PHYSLITE file length differs from the frozen identity")
    observed = REFERENCE / "full-clean-001"
    if json_file(observed / "verification.json")["source_file_uuid"] != source["tfile_uuid"]:
        raise ValueError("Stock observer is from a different original input")
    authority = {scope: str(REFERENCE / f"full-file-pilot-{stock}-002/01-full/histograms.npz")
                 for scope, stock in zip(SCOPES, ("small", "large", "combined"), strict=True)}
    for path in authority.values():
        Path(path).resolve(strict=True)

    options = ["--reuse-nominal-nnjvt-update"] if nnjvt_reuse else []
    sources = ("columnar/jets.py", "columnar/inputs.py", "jet_transport.py", "config/jet-states.json",
               "config/jet-histogram.json", "tools/compare_columnar.py", "tools/time_columnar_trials.py")
    source_identity = {str(ROOT / name): sha(ROOT / name) for name in sources}
    def arm(scopes, provider, data, runtime):
        payloads = {scope: data if isinstance(data, Path) else data[scope] for scope in scopes}
        return {
            **runtime, "provider": str(providers[provider]), "scopes": list(scopes), "options": options,
            "payloads": {scope: str(path) for scope, path in payloads.items()},
            "provenances": {scope: str(path.with_name("provenance.json")) for scope, path in payloads.items()},
        }

    regular, generic = build.regular, build.generic
    pairs = {
        "rng-size": (SCOPES, "bigger", original, regular, original),
        "variable-flow": (SCOPES, "flow", original, regular, original),
        "jmr-match": (("large", "both"), "match", original, regular, original),
        "generic-core": (SCOPES, "generic", original, generic, original),
        "packed-eta-jes": (SCOPES, "baseline", packed, regular, original),
        "packed-scope": (("small", "large"), "baseline", projected, regular, packed),
    }
    result.mkdir()
    for family in FAMILIES:
        scopes, provider, data, runtime, off_payload = pairs[family]
        emit(result / f"{family}.json", {
            "experiment": family, "build_label": label, "rounds": rounds,
            "input": str(input_path), "input_identity": source, "script": str(ROOT / "columnar/jets.py"),
            "script_sources": source_identity,
            "python": regular["python"], "scopes": list(scopes), "events": 20000,
            "observer": str(observed / "reference.root"), "authority": {key: authority[key] for key in scopes},
            "screen_events": screen_events, "screen_each_scope": True, "require_matching_branches": True,
            "compiler_only_runtime": True, "variants": {
                "off": arm(scopes, "baseline", off_payload, regular),
                "on": arm(scopes, provider, data, runtime),
            },
        })
    emit(result / "index.json", {
        "build_label": label, "build_receipts": str(build.work), "mode": "full",
        "families": {name: str(result / f"{name}.json") for name in FAMILIES},
        "nnjvt_update_reuse": nnjvt_reuse,
        "policy": "Independent observer gate before each full-file pair. Round one is warm-up when at least four rounds are requested.",
    })


if __name__ == "__main__":
    APP()
