"""Run the Path B framework timing through the stock, unobserved endpoint."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import typer


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_TOOLS = ROOT / "tools" / "reference"
sys.path.insert(0, str(REFERENCE_TOOLS))
from benchmark import (  # noqa: E402
    CPU, EVENTS, MODES, REPETITIONS, check_pair, check_quality, release_environment, run_leg,
)
from workflow import Scope, branch_states, open_root  # noqa: E402


app = typer.Typer(add_completion=False)
BUDGET = 1.20


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_library(leg, path, required):
    observed = {Path(item) for item in leg["libraries"] if Path(item).name == path.name}
    if (required and not observed) or observed - {path}:
        raise ValueError(f"Unexpected mapping for {path.name}: {sorted(map(str, observed))}")


def confirm_scope(directory, mode, scope, manifest, events):
    expected = {"sr": {"NOSYS"}, "lr": {"NOSYS"}}
    if mode == "full":
        expected = {"sr": set(manifest["sr_p4"]), "lr": set(manifest["lr_histogram"])}
    if scope == Scope.small:
        expected["lr"].clear()
    elif scope == Scope.large:
        expected["sr"].clear()
    with open_root(directory / "reference.root") as root_file:
        tree = root_file["analysis"]
        if tree.num_entries != events:
            raise ValueError(f"Incomplete framework sample in {directory}")
        for prefix, states in expected.items():
            if branch_states(set(tree.keys()), f"{prefix}_pt_") != states:
                raise ValueError(f"Unexpected {prefix} observable states in {directory}")
    executed = expected["sr"] | expected["lr"]
    if mode == "full" and expected["sr"]:
        executed.update(manifest["service_weight_only"])
    with open_root(directory / "hist-reference.root") as root_file:
        if set(root_file["systematicsList"].axis().labels()) != executed:
            raise ValueError(f"Unexpected service states in {directory}")
    if mode == "stored":
        if "Calibration sequence:" in (directory / "framework.log").read_text():
            raise ValueError(f"Stored control recalibrated jets in {directory}")


def compare_histograms(stock, overlay):
    with np.load(stock) as a, np.load(overlay) as b:
        if set(a.files) != set(b.files):
            raise ValueError("Stock and overlay histogram fields differ")
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


@app.command()
def run(result: Path, overlay: Path, provider: Path, payload: Path, sdk: Path,
        stock: Path | None = None, events: int = EVENTS, pilot: bool = False,
        scope: Scope = Scope.combined) -> None:
    """Measure the B framework leg and the matched separate histogram leg."""
    if os.sched_getaffinity(0) != {CPU} or len(list(Path("/proc/self/task").iterdir())) != 1:
        raise ValueError(f"Launch the B benchmark controller with one thread on CPU {CPU}")
    if not pilot and events != EVENTS:
        raise ValueError(f"The production endpoint uses {EVENTS} events")

    result, overlay, provider, payload, sdk = (path.resolve() for path in (result, overlay, provider, payload, sdk))
    core = (sdk / "lib/libcorrectionlib.so").resolve()
    libraries = {
        name: (overlay / f"lib{name}.so").resolve()
        for name in ("JetCalibToolsLib", "JetsCorrectionRuntime", "JetMomentToolsLib",
                     "JetJvtEfficiencyLib", "JetUncertaintiesLib", "ParticleJetToolsLib", "AsgAnalysisAlgorithmsLib")
    }
    source_paths = [*libraries.values(), provider, payload, core]
    source_hashes = {str(path): digest(path) for path in source_paths}
    environment = release_environment()
    environment["LD_LIBRARY_PATH"] = f"{overlay}:{sdk / 'lib'}:{environment['LD_LIBRARY_PATH']}"
    environment["JETS_PAYLOAD"] = str(payload)
    environment["JETS_PROVIDER"] = str(provider)
    environment.pop("JETS_FRAMEWORK_TRACE", None)
    release = Path(environment["AtlasArea"]) / "InstallArea" / environment["CMTCONFIG"]
    python = ROOT / ".reference-venv/bin/python"
    hist_env = dict(os.environ)
    for key in ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "JETS_FRAMEWORK_TRACE"):
        hist_env.pop(key, None)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "ROOT_MAX_THREADS"):
        hist_env[key] = "1"

    inventory = ROOT / "config/jet-states.json"
    manifest = json.loads(inventory.read_text())
    spec = ROOT / "config/jet-histogram.json"
    previous = {}
    if stock is not None:
        stock = stock.resolve()
        receipt = json.loads((stock / "provenance.json").read_text())
        if (receipt["events"], receipt["scope_collections"], receipt["input"]) != (
            events, scope.value, environment["JETS_REFERENCE_INPUT"],
        ):
            raise ValueError("Stock benchmark does not use the same input interval and collection scope")
        previous = json.loads((stock / "summary.json").read_text())
    result.mkdir()
    (result / "provenance.json").write_text(json.dumps({
        "path": "B", "input": environment["JETS_REFERENCE_INPUT"], "events": events,
        "cpu": CPU, "analysisbase": str(release), "scope_collections": scope.value,
        "warmup_count": 0 if pilot else 1, "artifact_sha256": source_hashes,
        "histogram_spec": str(spec), "histogram_spec_sha256": digest(spec),
        "stock": str(stock) if stock else None, "pilot": pilot,
        "clock": "monotonic wall from launch through exit, identical to the stock run_leg",
    }, indent=2) + "\n")

    measurements = {mode: defaultdict(list) for mode in MODES}
    hashes = {}
    for round_index in range(1 if pilot else REPETITIONS):
        outputs = {}
        for mode in MODES[round_index % len(MODES):] + MODES[:round_index % len(MODES)]:
            directory = result / f"{round_index + 1:02d}-{mode}"
            directory.mkdir()
            stem = "jets-stored" if mode == "stored" else "jets-a"
            suffix = "" if scope == Scope.combined else f"-{scope.value}"
            config = ROOT / "config" / f"{stem}{suffix}.yaml"
            command = [str(release / "bin/CPRun.py"), "-i", environment["JETS_REFERENCE_INPUT"],
                       "-t", str(config), "-o", "reference.root", "-e", str(events), "--direct-driver"]
            if mode == "nominal":
                command.append("--no-systematics")
            frame = run_leg(command, environment, directory, "framework")
            if mode != "stored":
                expected = {"JetCalibToolsLib", "JetsCorrectionRuntime", "JetUncertaintiesLib",
                            "ParticleJetToolsLib"}
                if scope != Scope.large:
                    expected.update(("JetMomentToolsLib", "JetJvtEfficiencyLib", "AsgAnalysisAlgorithmsLib"))
                if scope != Scope.small:
                    expected.add("AsgAnalysisAlgorithmsLib")
                for name, library in libraries.items():
                    verified_library(frame, library, name in expected)
                verified_library(frame, provider, True)
                verified_library(frame, core, True)
            confirm_scope(directory, mode, scope, manifest, events)
            outputs[mode] = directory / "histograms.npz"
            hist = run_leg([str(python), str(REFERENCE_TOOLS / "workflow.py"), "histogram",
                            str(directory / "reference.root"), str(inventory), str(spec),
                            str(outputs[mode]), mode, "--scope", scope.value], hist_env,
                           directory, "histogram")
            if stock:
                compare_histograms(stock / f"01-{mode}" / "histograms.npz", outputs[mode])
            current_hash = digest(outputs[mode])
            if mode in hashes and hashes[mode] != current_hash:
                raise ValueError(f"Non-reproducible B histogram in {directory}")
            hashes[mode] = current_hash
            measurements[mode]["framework"].append(frame)
            measurements[mode]["histogram"].append(hist)
            measurements[mode]["total"].append({"wall_s": frame["wall_s"] + hist["wall_s"]})
            print(f"{round_index + 1} {mode}: {frame['wall_s']:.3f}s + {hist['wall_s']:.3f}s", flush=True)
        check_pair(outputs["nominal"], outputs["full"])

    summary = {"runs": measurements, "histogram_sha256": hashes}
    if not pilot:
        quality = {mode: {leg: check_quality(records) for leg, records in metrics.items()}
                   for mode, metrics in measurements.items()}
        summary["quality"] = quality
        if "quality" in previous:
            a = previous["quality"]["full"]["framework"]
            b = quality["full"]["framework"]
            summary["framework_budget"] = {
                "ratio": b["mean_s"] / a["mean_s"], "limit": BUDGET,
                "passed": a["pass"] and b["pass"] and b["mean_s"] / a["mean_s"] <= BUDGET,
            }
    (result / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not pilot and not all(v["pass"] for legs in quality.values() for v in legs.values()):
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
