"""Run the matched A/B/C timing rounds using immutable per-job receipts."""
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Annotated
import zipfile

import numpy as np
import typer


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tools/reference"
sys.path[:0] = [str(REFERENCE), str(ROOT / "framework")]
from benchmark import (  # noqa: E402
    CPU, EVENTS, MAX_DEVIATION_FROM_MEDIAN, MODES, REPETITIONS,
    check_pair, check_quality, release_environment, run_leg,
)
from bench_path_b import BUDGET, confirm_scope, verified_library  # noqa: E402
from workflow import Scope, choose_names, open_root  # noqa: E402


APP = typer.Typer(add_completion=False)
INVENTORY = ROOT / "config/jet-states.json"
SPECIFICATION = ROOT / "config/jet-histogram.json"
HISTOGRAM_PYTHON = ROOT / ".reference-venv/bin/python"
THREAD_LIMITS = dict.fromkeys(
    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
     "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "RAYON_NUM_THREADS", "ROOT_MAX_THREADS"), "1",
)
FRAMEWORK_LIBRARIES = (
    "JetCalibToolsLib", "JetsCorrectionRuntime", "JetMomentToolsLib", "JetJvtEfficiencyLib",
    "JetUncertaintiesLib", "ParticleJetToolsLib", "AsgAnalysisAlgorithmsLib",
)
ROOT_LIBRARIES = (
    "libROOT", "libCore.so", "libRIO.so", "libHist.so", "libTree.so", "libCling.so",
    "libMathCore.so", "libPhysics.so", "libcppyy", "libCPyCppyy", "libxAOD",
    "libJetCalibTools", "libJetUncertainties", "libAsgTools", "liblwtnn.so",
)
ARM_ORDERS = ("ABC", "BAC", "CAB", "BCA")


class Cohort(str, Enum):
    faithful = "faithful"
    production = "production"


@dataclass(frozen=True)
class Job:
    round: int
    scope: Scope
    mode: str
    arm: str
    anchor: bool = False

    @property
    def key(self) -> str:
        if self.anchor:
            return f"anchor-{self.round:02d}-combined-full-A"
        return f"{self.round:02d}-{self.scope.value}-{self.mode}-{self.arm}"

    def details(self) -> dict:
        return {**self.__dict__, "scope": self.scope.value, "key": self.key,
                "warmup": self.round == 0 and not self.anchor}


def schedule() -> list[Job]:
    jobs = []
    groups = [(scope, mode) for scope in Scope for mode in MODES]
    for round_index in range(REPETITIONS):
        if round_index:
            jobs.append(Job(round_index, Scope.combined, "full", "A", anchor=True))
        for position, (scope, mode) in enumerate(groups[round_index:] + groups[:round_index]):
            for arm in ARM_ORDERS[(round_index + position) % len(ARM_ORDERS)]:
                jobs.append(Job(round_index, scope, mode, arm))
    jobs.append(Job(REPETITIONS, Scope.combined, "full", "A", anchor=True))
    return jobs


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sealed_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def selected_attempt(job_dir: Path):
    accepted = []
    for attempt in sorted(job_dir.glob("attempt-*")):
        receipt = attempt / "receipt.json"
        if receipt.is_file():
            contents = json.loads(receipt.read_text())
            if contents["classification"] in {"retained", "warmup", "anchor"}:
                if sha256(attempt / "histograms.npz") != contents["histogram_sha256"]:
                    raise ValueError(f"Histogram digest changed in {attempt}")
                for name, digest in contents["resource_receipts_sha256"].items():
                    if sha256(attempt / name) != digest:
                        raise ValueError(f"Resource receipt changed in {attempt / name}")
                accepted.append((attempt, contents))
    if len(accepted) > 1:
        raise ValueError(f"Conflicting accepted attempts in {job_dir}")
    return accepted[0] if accepted else None


def require_no_live_attempt(result: Path) -> None:
    running = []
    for proc in Path("/proc").glob("[0-9]*"):
        if proc.name == str(os.getpid()):
            continue
        try:
            cwd = (proc / "cwd").readlink()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if cwd.is_relative_to(result) and any(part.startswith("attempt-") for part in cwd.parts):
            running.append((proc.name, str(cwd)))
    if running:
        raise RuntimeError(f"An earlier campaign process is still running: {running}")


def exact_archive(actual: Path, expected: Path, projection: bool = False) -> None:
    with np.load(actual) as candidate, np.load(expected) as reference:
        require = set(candidate.files)
        expected_keys = set(reference.files)
        if (projection and not require <= expected_keys) or (not projection and require != expected_keys):
            raise ValueError(f"Histogram fields differ for {actual} and {expected}")
        for key in candidate.files:
            a, b = candidate[key], reference[key]
            if a.dtype != b.dtype or a.shape != b.shape:
                raise ValueError(f"Histogram type or shape differs in {key}: {actual} and {expected}")
            if a.tobytes() != b.tobytes():
                mismatches = np.flatnonzero(a.ravel() != b.ravel())
                index = np.unravel_index(int(mismatches[0]), a.shape) if len(mismatches) else ()
                raise ValueError(f"Histogram bytes differ in {key} at {index}: {actual} and {expected}")


def check_histogram(path: Path, job: Job, inventory: dict, specification: dict) -> None:
    small, large = choose_names(job.mode, inventory)
    names = {"sr": small, "lr": large}
    if job.scope == Scope.small:
        del names["lr"]
    elif job.scope == Scope.large:
        del names["sr"]
    with np.load(path) as data:
        expected = {"edges", "events"}
        for prefix in names:
            expected.update(f"{prefix}_{field}" for field in ("states", "entries", "sumw", "sumw2"))
        edges = np.asarray(specification["bins_gev"], dtype=np.float64)
        if set(data.files) != expected or int(data["events"]) != EVENTS:
            raise ValueError(f"Histogram scope or event interval differs in {path}")
        np.testing.assert_array_equal(data["edges"], edges)
        for prefix, states in names.items():
            if data[f"{prefix}_states"].tolist() != states:
                raise ValueError(f"Histogram states differ in {path}")
            if data[f"{prefix}_entries"].shape != (len(states),):
                raise ValueError(f"Histogram entries have an unexpected shape in {path}")
            for field in ("sumw", "sumw2"):
                bins = data[f"{prefix}_{field}"]
                if bins.dtype != np.float64 or bins.shape != (len(edges) + 1, len(states)):
                    raise ValueError(f"Histogram {prefix} {field} lacks expected flow bins in {path}")


def cpu_accounting(leg: dict) -> dict:
    before = leg["guest_before"]["cpu_stat"][f"cpu{CPU}"]
    after = leg["guest_after"]["cpu_stat"][f"cpu{CPU}"]
    ticks = [end - start for start, end in zip(before, after, strict=True)]
    # Linux guest counters are already included in user and nice.
    busy = sum(ticks[i] for i in (0, 1, 2, 5, 6, 7)) / os.sysconf("SC_CLK_TCK")
    measured = leg["cpu_user_s"] + leg["cpu_sys_s"] + leg["controller_cpu_s"]
    return {"cpu_stat_delta": ticks, "cpu_busy_s": busy, "measured_process_cpu_s": measured,
            "unattributed_busy_s": busy - measured}


def clean_environment() -> dict[str, str]:
    result = dict(os.environ)
    for key in ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD",
                "JETS_FRAMEWORK_TRACE", "ATLAS_JET_LIBRARY_AUDIT", "VIRTUAL_ENV"):
        result.pop(key, None)
    result.update(THREAD_LIMITS)
    return result


def verify_framework(job: Job, attempt: Path, receipt: dict, inventory: dict,
                     libraries: dict, b_provider: Path, core: Path) -> None:
    confirm_scope(attempt, job.mode, job.scope, inventory, EVENTS)
    with open_root(attempt / "hist-reference.root") as counters:
        labels = counters["systematicsList"].axis().labels()
        expected_order = [state for state in inventory["service_kinematic"] if state in set(labels)]
        observed_order = [state for state in labels if state not in inventory["service_weight_only"]]
        if expected_order != observed_order:
            raise ValueError(f"Kinematic service state order differs in {attempt}")
    if job.arm == "A":
        mappings = {Path(path) for path in receipt["libraries"]}
        if mappings & ({*libraries.values(), b_provider, core}):
            raise ValueError(f"Stock path loaded a converted library in {attempt}")
        return
    required = set()
    if job.mode != "stored":
        required = {"JetCalibToolsLib", "JetsCorrectionRuntime", "JetUncertaintiesLib",
                    "ParticleJetToolsLib", "AsgAnalysisAlgorithmsLib"}
        if job.scope != Scope.large:
            required.update(("JetMomentToolsLib", "JetJvtEfficiencyLib"))
    for stem, expected in libraries.items():
        verified_library(receipt, expected, stem in required)
    verified_library(receipt, b_provider, bool(required))
    verified_library(receipt, core, bool(required))


def verify_columnar(job: Job, attempt: Path, receipt: dict, inventory: dict,
                    source: dict, cohort: Cohort, core: Path, provider: Path | None,
                    wheel_members: dict) -> dict:
    log = (attempt / "columnar.log").read_text()
    summary = f"{EVENTS} unique events, {EVENTS} event visits across 1 passes"
    if log.count(summary) != 1 or log.count(f"Pass 1/1, {EVENTS} events") != 1:
        raise ValueError(f"Columnar process did not make exactly one complete input pass in {attempt}")
    if log.count(f"({source['source_file_uuid']})") != 1:
        raise ValueError(f"Columnar source UUID differs in {attempt}")
    selected = {"small": {"sr"}, "large": {"lr"}, "combined": {"sr", "lr"}}[job.scope.value]
    rows = re.findall(r"(?m)^(sr|lr) (\d+) unique jets, (\d+) jet visits, (\d+) nominal entries, (\d+) states$", log)
    if {row[0] for row in rows} != selected or len(rows) != len(selected):
        raise ValueError(f"Columnar visit counters differ in {attempt}")
    items = {}
    for prefix, jets, visits, counted, states in rows:
        expected_jets = source["collections"][prefix]["objects"]
        expected_states = len(inventory[f"{prefix}_histogram"]) if job.mode == "full" else 1
        if int(jets) != expected_jets or int(visits) != expected_jets or int(states) != expected_states:
            raise ValueError(f"Unexpected {prefix} columnar counters in {attempt}")
        items[prefix] = {"jets": int(jets), "visits": int(visits), "nominal_entries": int(counted),
                         "states": int(states)}
    if cohort == Cohort.faithful and job.mode != "stored":
        verified_library(receipt, provider, True)
        verified_library(receipt, core, True)
    if cohort == Cohort.production:
        loaded = {Path(path) for path in receipt["libraries"]}
        runtime_audit = attempt / "wheel-runtime.json"
        if runtime_audit.is_file():
            audited = json.loads(runtime_audit.read_text())
            loaded.update(Path(path) for path in audited["libraries"])
        if any(path.is_relative_to("/cvmfs") or path.name.startswith(ROOT_LIBRARIES) for path in loaded):
            raise ValueError(f"Production wheel loaded a release/ROOT library in {attempt}")
        for name, digest in wheel_members["native"].items():
            matches = [path for path in loaded if path.name == name]
            required = job.mode != "stored" or name in {"libstdc++.so.6", "libgcc_s.so.1"}
            if (required and len(matches) != 1) or any(sha256(path) != digest for path in matches):
                raise ValueError(f"Production wheel runtime differs for {name} in {attempt}")
        compiler, = (path for path in loaded if path.name == "libstdc++.so.6")
        data = compiler.parents[2] / "atlas_jet_demo/data"
        for name, digest in wheel_members["data"].items():
            if sha256(data / name) != digest:
                raise ValueError(f"Installed production payload differs for {name} in {attempt}")
    return items


@dataclass(frozen=True)
class Context:
    cohort: Cohort
    input_file: Path
    source: dict
    framework_python: Path
    framework_env: dict
    overlay_env: dict
    columnar_env: dict
    libraries: dict
    b_provider: Path
    core: Path
    columnar_python: Path
    columnar_script: Path
    c_provider: Path | None
    c_payload: Path | None
    wheel_members: dict


def execute(job: Job, attempt: Path, context: Context, inventory: dict,
            specification: dict, diagnostic: bool) -> dict:
    output = attempt / "histograms.npz"
    if job.arm == "C":
        scope = "both" if job.scope == Scope.combined else job.scope.value
        command = [
            str(context.columnar_python), str(context.columnar_script), str(context.input_file),
            str(output), "--mode", job.mode, "--scope", scope, "--repeat", "1", "--events", str(EVENTS),
        ]
        if context.cohort == Cohort.faithful:
            command.extend(("--states", str(INVENTORY), "--histogram-spec", str(SPECIFICATION)))
            if job.mode != "stored":
                command.extend(("--provider", str(context.c_provider), "--payload", str(context.c_payload)))
        environment = dict(context.columnar_env)
        if diagnostic and context.cohort == Cohort.production:
            environment["ATLAS_JET_LIBRARY_AUDIT"] = str(attempt / "wheel-runtime.json")
        leg = run_leg(command, environment, attempt, "columnar")
        counters = verify_columnar(
            job, attempt, leg, inventory, context.source, context.cohort,
            context.core, context.c_provider, context.wheel_members,
        )
        times = {"total": leg["wall_s"]}
        accounting = {"columnar": cpu_accounting(leg)}
    else:
        stem = "jets-stored" if job.mode == "stored" else "jets-a"
        suffix = "" if job.scope == Scope.combined else f"-{job.scope.value}"
        config = ROOT / "config" / f"{stem}{suffix}.yaml"
        command = [
            str(context.framework_python), "-i", str(context.input_file), "-t", str(config),
            "-o", "reference.root", "-e", str(EVENTS), "--direct-driver",
        ]
        if job.mode == "nominal":
            command.append("--no-systematics")
        environment = context.framework_env if job.arm == "A" else context.overlay_env
        frame = run_leg(command, environment, attempt, "framework")
        verify_framework(job, attempt, frame, inventory, context.libraries, context.b_provider, context.core)
        hist = run_leg(
            [str(HISTOGRAM_PYTHON), str(REFERENCE / "workflow.py"), "histogram",
             str(attempt / "reference.root"), str(INVENTORY), str(SPECIFICATION),
             str(output), job.mode, "--scope", job.scope.value],
            clean_environment(), attempt, "histogram",
        )
        counters = {}
        times = {"framework": frame["wall_s"], "histogram": hist["wall_s"],
                 "total": frame["wall_s"] + hist["wall_s"]}
        accounting = {"framework": cpu_accounting(frame), "histogram": cpu_accounting(hist)}
    check_histogram(output, job, inventory, specification)
    classification = "anchor" if job.anchor else ("warmup" if job.details()["warmup"] else "retained")
    return {
        "job": job.details(),
        "classification": "diagnostic" if diagnostic else classification,
        "histogram_sha256": sha256(output),
        "resource_receipts_sha256": {path.name: sha256(path) for path in attempt.glob("*.json")
                                     if path.name in {"framework.json", "histogram.json", "columnar.json"}},
        "wall_s": times,
        "cpu_accounting": accounting,
        "input_visit_counters": counters,
    }


def check_campaign_histograms(jobs: dict[Job, tuple]) -> int:
    outputs = {job: attempt / "histograms.npz" for job, (attempt, _) in jobs.items()}
    compared = 0
    for job, result in outputs.items():
        peers = []
        projection = None
        if job.anchor:
            peers.append(Job(0, Scope.combined, "full", "A"))
        else:
            if job.arm != "A":
                peers.append(Job(job.round, job.scope, job.mode, "A"))
            if job.round:
                peers.append(Job(0, job.scope, job.mode, job.arm))
            if job.scope != Scope.combined:
                projection = Job(job.round, Scope.combined, job.mode, job.arm)
            if job.mode == "nominal":
                full = outputs.get(Job(job.round, job.scope, "full", job.arm))
                if full:
                    check_pair(result, full)
                    compared += 1
        for peer in peers:
            if peer in outputs:
                exact_archive(result, outputs[peer])
                compared += 1
        if projection in outputs:
            exact_archive(result, outputs[projection], projection=True)
            compared += 1
    return compared


def campaign_summary(jobs: dict[Job, tuple]) -> dict:
    comparisons = check_campaign_histograms(jobs)
    summary = {"completed": len(jobs), "planned": len(schedule()), "histogram_guards": comparisons,
               "progress": {job.key: str(attempt) for job, (attempt, _) in jobs.items()}}
    if len(jobs) != len(schedule()):
        return summary
    measurements = defaultdict(dict)
    for scope in Scope:
        for mode in MODES:
            for arm in "ABC":
                receipts = [jobs[Job(r, scope, mode, arm)][1]["wall_s"] for r in range(REPETITIONS)]
                measurements[scope.value][f"{mode}-{arm}"] = {
                    leg: check_quality([{"wall_s": row[leg]} for row in receipts]) for leg in receipts[0]
                }
    anchors = [jobs[job][1]["wall_s"] for job in schedule() if job.anchor]
    drift = {}
    for leg in anchors[0]:
        values = [anchor[leg] for anchor in anchors]
        median = statistics.median(values)
        deviation = max(abs(v - median) for v in values) / median
        drift[leg] = {"wall_s": values, "median_s": median, "max_median_deviation": deviation,
                      "pass": deviation <= MAX_DEVIATION_FROM_MEDIAN}
    ratios = {}
    for scope, modes in measurements.items():
        ratios[scope] = {}
        for mode in MODES:
            a, b, c = (modes[f"{mode}-{arm}"] for arm in "ABC")
            applicable = all(leg["pass"] for arm in (a, b, c) for leg in arm.values())
            profiling = all(row["pass"] for row in drift.values())
            ratio = b["framework"]["mean_s"] / a["framework"]["mean_s"]
            framework_quality = profiling and a["framework"]["pass"] and b["framework"]["pass"]
            ratios[scope][mode] = {
                "quality_pass": applicable and profiling,
                "B_over_A_framework": ratio,
                "B_framework_budget_limit": BUDGET,
                "B_framework_budget_pass": ratio <= BUDGET if framework_quality else None,
                "A_over_C_end_to_end": a["total"]["mean_s"] / c["total"]["mean_s"],
                "B_over_C_end_to_end": b["total"]["mean_s"] / c["total"]["mean_s"],
            }
    summary.update(quality=measurements, stock_anchor_drift=drift, ratios=ratios)
    summary["quality_pass"] = all(value["quality_pass"] for scope in ratios.values() for value in scope.values())
    return summary


def installed_wheel(wheel: Path | None, cohort: Cohort) -> dict:
    if cohort == Cohort.faithful:
        if wheel is not None:
            raise ValueError("Faithful baseline uses the release-closure columnar runtime")
        return {}
    if wheel is None:
        raise ValueError("Production requires the installed demonstration wheel")
    expected = {"libcorrectionlib.so", "libjets.so", "libgcc_s.so.1", "libstdc++.so.6"}
    with zipfile.ZipFile(wheel) as archive:
        members = {}
        for item in archive.namelist():
            name = Path(item).name
            if name in expected or (name.startswith("_core.") and name.endswith(".so")):
                if name in members:
                    raise ValueError(f"Duplicate native wheel member {name}")
                members[name] = hashlib.sha256(archive.read(item)).hexdigest()
        if not expected <= members.keys() or len(members) != len(expected) + 1:
            raise ValueError(f"Demonstration wheel native members differ: {list(members)}")
        data = {
            name: hashlib.sha256(archive.read(f"atlas_jet_demo/data/{name}")).hexdigest()
            for name in ("jets.json", "jet-states.json", "jet-histogram.json", "provenance.json", "build.json")
        }
    if data["jet-states.json"] != sha256(INVENTORY) or data["jet-histogram.json"] != sha256(SPECIFICATION):
        raise ValueError("Production wheel uses a different state inventory or histogram specification")
    return {"native": members, "data": data}


def verify_acceptance(b: Path | None, c: Path | None, diagnostic: bool, overlay: Path,
                      provider: Path, payload: Path, core: Path) -> dict:
    proofs = {}
    if not diagnostic and (b is None or c is None):
        raise ValueError("Supply accepted full-file observer receipts for both converted paths")
    if b:
        proof = json.loads(b.read_text())
        if (proof["events"] != EVENTS or proof["branches"] != 7189 or proof["service_states"] != 331
                or not proof["exact_observer_parity"] or not proof["exact_histogram_parity"]):
            raise ValueError(f"Incomplete or failing full framework verification in {b}")
        hashes = dict(reversed(row.split(None, 1)) for row in
                      (b.parent / "runtime-inputs.sha256").read_text().splitlines())
        for path in (provider, payload, core, *sorted(overlay.glob("*.so"))):
            if hashes.get(str(path)) != sha256(path):
                raise ValueError(f"Framework verification does not cover {path}")
        proofs["B"] = {"path": str(b), "sha256": sha256(b), "authority": proof["reference"]}
    if c:
        proof = json.loads(c.read_text())
        if (proof["entries_compared"] != EVENTS or proof["scope"] != "both" or proof["mode"] != "full"
                or proof["stage_limit"] != "final" or not proof["passed"]):
            raise ValueError(f"Incomplete or failing direct PHYSLITE verification in {c}")
        proofs["C"] = {"path": str(c), "sha256": sha256(c), "authority": proof["authority"],
                       "provider": proof["provider"], "payload": proof["payload"],
                       "core": proof["runtime"]["core"]}
    return proofs


@APP.command()
def plan() -> None:
    """Show the cross-path schedule, including independent stock drift anchors."""
    typer.echo(json.dumps([job.details() for job in schedule()], indent=2))


@APP.command()
def run(
    result: Path,
    cohort: Annotated[Cohort, typer.Option()],
    overlay: Annotated[Path, typer.Option()],
    provider: Annotated[Path, typer.Option(help="Provider loaded by the framework overlay")],
    payload: Annotated[Path, typer.Option(help="Payload loaded by the framework overlay")],
    sdk: Annotated[Path, typer.Option()],
    columnar_python: Annotated[Path, typer.Option(help="Python from the prepared columnar environment")],
    columnar_script: Annotated[Path, typer.Option(help="Faithful source CLI or production visible demo")],
    source_verification: Annotated[Path, typer.Option(help="Clean stock observer coverage receipt")],
    b_verification: Annotated[Path | None, typer.Option()] = None,
    c_verification: Annotated[Path | None, typer.Option()] = None,
    wheel: Annotated[Path | None, typer.Option(help="Exact production demo wheel beside its visible script")] = None,
    columnar_provider: Annotated[Path | None, typer.Option()] = None,
    columnar_payload: Annotated[Path | None, typer.Option()] = None,
    jobs: Annotated[int | None, typer.Option(min=1, help="Stop after this many new jobs")] = None,
    diagnostic: Annotated[bool, typer.Option(help="Record visits without including them in timed cohorts")] = False,
) -> None:
    """Run or resume one exact full-file cohort on the reserved CPU."""
    if os.sched_getaffinity(0) != {CPU} or len(list(Path("/proc/self/task").iterdir())) != 1:
        raise typer.BadParameter(f"Launch the controller with one thread on CPU {CPU}")
    if diagnostic and jobs is None:
        raise typer.BadParameter("Set a job limit for a diagnostic invocation")

    result = result.resolve()
    overlay, provider, payload, sdk, columnar_script, source_verification = (
        path.resolve(strict=True) for path in (overlay, provider, payload, sdk, columnar_script, source_verification)
    )
    # Keep the venv executable's path, so CPython finds its pyvenv.cfg.
    columnar_python = columnar_python.absolute()
    if not columnar_python.is_file():
        raise FileNotFoundError(columnar_python)
    core = (sdk / "lib/libcorrectionlib.so").resolve(strict=True)
    libraries = {name: (overlay / f"lib{name}.so").resolve(strict=True) for name in FRAMEWORK_LIBRARIES}
    b_verification, c_verification, wheel, columnar_provider, columnar_payload = (
        path.resolve(strict=True) if path else None
        for path in (b_verification, c_verification, wheel, columnar_provider, columnar_payload)
    )
    if cohort == Cohort.faithful and not (columnar_provider and columnar_payload):
        raise typer.BadParameter("Faithful C requires a provider and payload")
    if cohort == Cohort.production and (columnar_provider or columnar_payload):
        raise typer.BadParameter("Production C loads the executable assets inside its wheel")
    wheel_members = installed_wheel(wheel, cohort)
    wheel_proof = None
    if wheel:
        wheel_proof = wheel.with_name("bundle.json")
        bundle = json.loads(wheel_proof.read_text())
        if (wheel.parent != columnar_script.parent or bundle["wheel"]["sha256"] != sha256(wheel)
                or bundle["visible_script"]["sha256"] != sha256(columnar_script)
                or bundle["provider"]["sha256"] != sha256(provider)
                or bundle["payload"]["sha256"] != sha256(payload)
                or wheel_members["data"]["jets.json"] != sha256(payload)):
            raise ValueError("Production framework, visible script, or wheel differs from the bundle receipt")
    if cohort == Cohort.faithful and (sha256(provider) != sha256(columnar_provider)
                                     or sha256(payload) != sha256(columnar_payload)):
        raise ValueError("Faithful paths B and C must use the same native provider and payload")

    proofs = verify_acceptance(b_verification, c_verification, diagnostic, overlay, provider, payload, core)
    if c_verification:
        c_source = Path(proofs["C"]["provider"])
        c_core = Path(proofs["C"]["core"])
        native_digests = wheel_members.get("native", {})
        same_provider = sha256(c_source) in {sha256(provider), native_digests.get("libjets.so")}
        same_core = sha256(c_core) in {sha256(core), native_digests.get("libcorrectionlib.so")}
        if not same_provider or not same_core or sha256(Path(proofs["C"]["payload"])) != sha256(payload):
            raise ValueError("Columnar acceptance does not cover this provider, payload, and evaluator")
    independent_stock = source_verification.with_name("reference.root")
    if any(Path(proof["authority"]).resolve() != independent_stock for proof in proofs.values()):
        raise ValueError("Converted observers did not use the clean stock reference")
    source = json.loads(source_verification.read_text())
    input_spec = json.loads((ROOT / "config/jets-input.json").read_text())
    input_file = Path(input_spec["path"])
    if (source["events"] != EVENTS or source["source_file_uuid"] != input_spec["tfile_uuid"]
            or sha256(input_file) != input_spec["sha256"]):
        raise ValueError("The clean stock verification or original PHYSLITE file differs from the frozen sample")

    framework_env = release_environment()
    for key in ("JETS_PROVIDER", "JETS_PAYLOAD", "JETS_FRAMEWORK_TRACE", "ATLAS_JET_LIBRARY_AUDIT"):
        framework_env.pop(key, None)
    framework_env.update(THREAD_LIMITS)
    if Path(framework_env["JETS_REFERENCE_INPUT"]).resolve() != input_file.resolve():
        raise ValueError("Stock release resolved a different input PHYSLITE file")
    release = Path(framework_env["AtlasArea"]) / "InstallArea" / framework_env["CMTCONFIG"]
    framework_python = release / "bin/CPRun.py"
    overlay_env = {
        **framework_env,
        "LD_LIBRARY_PATH": f"{overlay}:{sdk / 'lib'}:{framework_env['LD_LIBRARY_PATH']}",
        "JETS_PROVIDER": str(provider), "JETS_PAYLOAD": str(payload),
    }
    if any(path in framework_env["LD_LIBRARY_PATH"].split(":") for path in (str(overlay), str(sdk / "lib"))):
        raise ValueError("Start the timing controller outside the converted framework environment")
    columnar_env = clean_environment()
    if cohort == Cohort.faithful:
        columnar_env["LD_LIBRARY_PATH"] = f"{sdk / 'lib'}:{framework_env['LD_LIBRARY_PATH']}"

    inventory = json.loads(INVENTORY.read_text())
    specification = json.loads(SPECIFICATION.read_text())
    identity_paths = [
        ROOT / "tools/benchmark_campaign.py", REFERENCE / "benchmark.py", REFERENCE / "workflow.py",
        REFERENCE / "setup.sh", ROOT / "framework/bench_path_b.py", INVENTORY, SPECIFICATION,
        source_verification, provider, payload, core, columnar_python, columnar_script,
        *libraries.values(), *sorted((ROOT / "config").glob("jets-a*.yaml")),
        *sorted((ROOT / "config").glob("jets-stored*.yaml")),
    ]
    identity_paths.extend(path for path in (wheel, wheel_proof, columnar_payload, columnar_provider) if path)
    scheduled = schedule()
    provenance = {
        "cohort": cohort.value, "cpu": CPU, "thread_limits": THREAD_LIMITS, "release": str(release),
        "guest_node": os.uname().nodename, "guest_release": os.uname().release,
        "original": input_spec, "accepted_observers": proofs,
        "identity_sha256": {str(path): sha256(path) for path in identity_paths},
        "wheel_members_sha256": wheel_members,
        "endpoint": {"A": ["CPRun", "histogram"], "B": ["CPRun", "histogram"],
                     "C": ["one direct PHYSLITE histogram pass"]},
        "endpoint_clock": "monotonic wall including startup and output serialization",
        "schedule": [job.details() for job in scheduled],
    }
    context = Context(
        cohort, input_file, source, framework_python, framework_env, overlay_env, columnar_env,
        libraries, provider, core, columnar_python, columnar_script, columnar_provider,
        columnar_payload, wheel_members,
    )

    with (ROOT / f"results/.benchmark-cpu{CPU}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result.mkdir(parents=True, exist_ok=True)
        require_no_live_attempt(result)
        campaign = result / "campaign.json"
        if campaign.is_file():
            if json.loads(campaign.read_text()) != provenance:
                raise ValueError(f"Configuration or artifact identity changed for {result}")
        else:
            sealed_json(campaign, provenance)
        finished = {
            job: accepted for job in scheduled
            if (accepted := selected_attempt(result / job.key)) is not None
        }
        sealed_json(result / "summary.json", campaign_summary(finished))
        started = 0
        for job in scheduled:
            if job in finished:
                continue
            if jobs is not None and started >= jobs:
                break
            directory = result / job.key
            directory.mkdir(exist_ok=True)
            attempt = directory / f"attempt-{len(list(directory.glob('attempt-*'))) + 1:03d}"
            attempt.mkdir()
            started += 1
            typer.echo(f"{cohort.value} {job.key} {attempt.name}", err=True)
            launch = {"unix_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
                      "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
            sealed_json(attempt / "job.json", {**job.details(), "launch": launch})
            try:
                receipt = execute(job, attempt, context, inventory, specification, diagnostic)
            except Exception as error:
                sealed_json(attempt / "receipt.json", {
                    "job": job.details(), "classification": "diagnostic", "failure": str(error),
                    "launch": launch, "completed_unix_ns": time.time_ns(),
                })
                raise
            receipt.update(launch=launch, completed_unix_ns=time.time_ns())
            sealed_json(attempt / "receipt.json", receipt)
            if not diagnostic:
                finished[job] = (attempt, receipt)
                sealed_json(result / "summary.json", campaign_summary(finished))
            typer.echo(f"  {receipt['wall_s']['total']:.3f} s ({receipt['classification']})", err=True)
        summary = campaign_summary(finished)
        typer.echo(f"{summary['completed']} of {summary['planned']} jobs accepted in {result}")
        if summary.get("quality_pass") is False:
            raise typer.Exit(2)


if __name__ == "__main__":
    APP()
