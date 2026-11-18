"""Freeze and verify the MT19937 prefix pair around the shared C trial runner."""

from pathlib import Path

import typer

from tools.benchmark_campaign import exact_archive
from tools.joint_trials import (
    BUILD, COMPILER_LIB, EXPORT, REFERENCE, ROOT, SCOPES, checksums, emit, json_file, sha, verify_payload,
)
from tools.profile_final_demo import stock_endpoint


APP = typer.Typer(add_completion=False)
ARMS = ("off", "on")
PACKED = EXPORT / "eta-jes-packed-001/jets.json"
BUNDLE = ROOT / "provenance/provider-patches/rng-prefix"


def seal_checksums(output, paths):
    with output.open("x") as stream:
        for path in dict.fromkeys(paths):
            stream.write(f"{sha(path)}  {path}\n")


def observe_stamp(path):
    record = path.stat()
    return [record.st_dev, record.st_ino, record.st_size, record.st_mtime_ns, record.st_ctime_ns]


def verify_build_pair(tag):
    sdk = BUILD / "core-precision-002/venv/lib/python3.14/site-packages/correctionlib"
    python = BUILD / "core-precision-002/venv/bin/python"
    core = sdk / "lib/libcorrectionlib.so"
    sources, variants, revisions, flags, inputs = {}, {}, {}, {}, [core, python]
    if not python.is_file():
        raise FileNotFoundError(python)
    for arm in ARMS:
        directory = BUILD / f"rng-prefix-{arm}-{tag}"
        choices = {"rng-cache-large", "fast-variable-flow"}
        defines = {"-DJETS_PORTABLE", "-DJETS_RNG_CACHE", "-DJETS_RNG_CACHE_LARGE", "-DJETS_FAST_VARIABLE_FLOW"}
        if arm == "on":
            choices.add("rng-prefix")
            defines.add("-DJETS_RNG_PREFIX")
        if ((directory / "exit-code").read_text().strip() != "0"
                or set((directory / "build-choices.txt").read_text().splitlines()) != choices
                or set((directory / "build-defines.txt").read_text().splitlines()) != defines
                or (directory / "rng-cache-slots.txt").read_text().strip() != "262144"):
            raise ValueError(f"Unexpected flags or status for {directory}")
        provider = directory / "libjets.so"
        native = checksums(directory / "native.sha256")
        for file, digest in native.items():
            if sha(file) != digest:
                raise ValueError(f"Native build bytes changed: {file}")
        if {core, provider} - native.keys():
            raise ValueError(f"Missing core/provider receipt in {directory}")
        for launch, digest in checksums(directory / "launched-script.sha256").items():
            if sha(launch) != digest:
                raise ValueError(f"Frozen build script changed: {launch}")
        sources[arm] = checksums(directory / "source-inputs.sha256")
        revisions[arm] = (directory / "source-revision.txt").read_text().strip()
        flags[arm] = (directory / "compile-flags.txt").read_text().splitlines()
        variants[arm] = {"python": str(python), "sdk": str(sdk), "provider": str(provider),
                         "payloads": dict.fromkeys(SCOPES, str(PACKED)),
                         "provenances": dict.fromkeys(SCOPES, str(PACKED.with_name("provenance.json"))),
                         "options": ["--reuse-nominal-nnjvt-update"]}
        inputs.extend(directory / name for name in ("exit-code", "libjets.so", "native.sha256", "build-choices.txt",
                                                    "build-defines.txt", "compile-flags.txt", "rng-cache-slots.txt",
                                                    "source-inputs.sha256", "source-revision.txt", "launched-script.sh"))
        inputs.extend(native)
    if (sources["off"] != sources["on"] or revisions["off"] != revisions["on"]
            or flags["on"].count("-DJETS_RNG_PREFIX") != 1
            or flags["off"] != [flag for flag in flags["on"] if flag != "-DJETS_RNG_PREFIX"]
            or sha(Path(variants["off"]["provider"])) == sha(Path(variants["on"]["provider"]))):
        raise ValueError("RNG prefix pair must use one source and differ only by the prefix define")
    for path, digest in sources["off"].items():
        if sha(path) != digest:
            raise ValueError(f"Provider source drifted after the paired builds: {path}")
    return variants, revisions["off"], inputs


@APP.command()
def freeze(tag: str, output: Path, screen_events: int):
    variants, revision, binary_inputs = verify_build_pair(tag)
    provenance = PACKED.with_name("provenance.json")
    packing_file = PACKED.with_name("packing.json")
    partition_file = EXPORT / "eta-jes-partition-001/manifest.json"
    packing, partition = map(json_file, (packing_file, partition_file))
    verify_payload(PACKED, packing["packed"])
    verify_payload(PACKED, partition["canonical"])
    if (json_file(provenance)["sha256"] != packing["packed"]["sha256"]
            or sha(provenance) != partition["provenance"]["sha256"]):
        raise ValueError("Packed payload provenance contradicts its sealed manifests")

    source_path = ROOT / "config/jets-input.json"
    source = json_file(source_path)
    original = Path(source["path"])
    verify_payload(original, source)
    observed = REFERENCE / "full-clean-001"
    observer = observed / "reference.root"
    observer_receipt = observed / "verification.json"
    observer_hash = observed / "reference.sha256"
    observed_digest = observer_hash.read_text().split()[0]
    coverage = json_file(observer_receipt)
    stamp = observe_stamp(observer)
    if (coverage["source_file_uuid"] != source["tfile_uuid"] or coverage["events"] != 20000
            or sha(observer) != observed_digest or observe_stamp(observer) != stamp):
        raise ValueError("Independent observer changed or covers a different input interval")

    scopes = dict(zip(SCOPES, ("small", "large", "combined"), strict=True))
    authorities = {
        scope: stock_endpoint(REFERENCE / f"full-file-pilot-{prefix}-002", scope, original, source, sha)
        for scope, prefix in scopes.items()
    }
    script_sources = [
        ROOT / name for name in (
            "columnar/jets.py", "columnar/inputs.py", "jet_transport.py", "config/jet-states.json",
            "config/jet-histogram.json", "tools/compare_columnar.py", "tools/time_columnar_trials.py",
            "tools/screen_trials.sh", "tools/qualify_rng_prefix.sh", "tools/prepare_rng_prefix_pair.py",
            "tools/benchmark_campaign.py", "tools/profile_final_demo.py", "tools/joint_trials.py",
            "tools/reference/setup.sh",
        )
    ]
    plan = {
        "experiment": "rng-prefix", "build_label": tag, "build_revision": revision,
        "input": str(original), "input_identity": source, "scopes": list(SCOPES), "events": 20000,
        "screen_events": screen_events, "script": str(ROOT / "columnar/jets.py"),
        "script_sources": {str(path): sha(path) for path in script_sources},
        "observer": str(observer), "observer_stamp": stamp, "observer_sha256": observed_digest,
        "authority": {scope: authority["histogram"]["path"] for scope, authority in authorities.items()},
        "stock_evidence": authorities, "screen_each_scope": True, "require_matching_branches": True,
        "compiler_only_runtime": True, "variants": variants, "rounds": 1,
    }
    emit(output / "screen.json", plan)
    stock_files = [
        Path(entry["path"]) for authority in authorities.values()
        for entry in (authority["histogram"], authority["provenance"], *authority["legs"].values())
    ]
    seal_checksums(
        output / "inputs.sha256",
        (output / "screen.json", source_path, original, observer_receipt, observer_hash, PACKED, packing_file,
         partition_file, provenance, BUNDLE / "integration.diff", COMPILER_LIB / "libstdc++.so.6",
         COMPILER_LIB / "libgcc_s.so.1", *stock_files, *binary_inputs, *script_sources),
    )


@APP.command()
def check_observer(plan: Path):
    recorded = json_file(plan)
    if observe_stamp(Path(recorded["observer"])) != recorded["observer_stamp"]:
        raise ValueError("Independent observer changed since the pair was frozen")


def checked_workflow(plan, workflow):
    origin = workflow / "inputs.json"
    summary = workflow / "summary.json"
    if json_file(origin)["plan"] != plan:
        raise ValueError(f"Executed trial plan differs in {origin}")
    expected = {(r, scope, arm) for r in range(1, plan["rounds"] + 1) for scope in SCOPES for arm in ARMS}
    report = json_file(summary)
    if report["observations"] != len(expected) or not report["all_histograms_equal"]:
        raise ValueError(f"Incomplete exact trial in {workflow}")
    files, observed = [origin, summary], set()
    for receipt in sorted(workflow.glob("*/receipt.json")):
        row = json_file(receipt)
        key = row["round"], row["scope"], row["name"]
        if key not in expected or key in observed:
            raise ValueError(f"Unexpected workflow case in {receipt}")
        if row["exit"] or not (row["exact_histogram"] and row["process"]["passed"] and row["matching_source_branches"]):
            raise ValueError(f"Unaccepted workflow case in {receipt}")
        histogram = receipt.with_name("histogram.npz")
        exact_archive(histogram, Path(plan["authority"][row["scope"]]))
        observed.add(key)
        files.extend((receipt, histogram, receipt.with_name("run.log")))
    if observed != expected:
        raise ValueError(f"Incomplete set of workflow receipts in {workflow}")
    return files


@APP.command()
def seal_screen(output: Path):
    plan = json_file(output / "screen.json")
    pair = output / "screen/pair"
    proof = pair / "object-screen.json"
    objects = json_file(proof)
    cases = objects["cases"]
    expected = {(scope, arm) for scope in SCOPES for arm in ARMS}
    if (not objects["passed"] or len(cases) != len(expected)
            or {(case["scope"], case["variant"]) for case in cases} != expected
            or objects["entries_per_case"] != plan["screen_events"]):
        raise ValueError("Incomplete independent object gate for the first screen")
    workflow = checked_workflow(plan, pair / "workflow")
    # Only expose the four-round plan after all six original-file screen jobs
    # and all six independent object cases have been checked.
    confirm = output / "confirm.json"
    emit(confirm, {**plan, "rounds": 4})
    seal_checksums(output / "screen-evidence.sha256",
                   (confirm, proof, pair / "exit-code", output / "screen/fixture-logs.sha256",
                    *(Path(case["proof"]) for case in cases), *workflow))


@APP.command()
def check_confirmation(output: Path):
    plan = json_file(output / "confirm.json")
    if plan["rounds"] != 4:
        raise ValueError("Confirmation must contain one excluded and three retained rounds")
    files = checked_workflow(plan, output / "confirm/workflow")
    seal_checksums(output / "confirm/workflow-evidence.sha256", files)


if __name__ == "__main__":
    APP()
