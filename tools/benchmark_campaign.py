"""Run the matched A/B/C timing rounds using immutable per-job receipts."""
import ast
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Annotated
import zipfile

import numpy as np
import typer


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tools/reference"
sys.path[:0] = [str(REFERENCE), str(ROOT / "framework")]
from benchmark import (  # noqa: E402
    CPU, EVENTS, MAX_DEVIATION_FROM_MEDIAN, MODES, REPETITIONS, STARTUP_LIBRARY_OBSERVATIONS,
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
FAITHFUL_C_FILES = ("columnar/jets.py", "columnar/inputs.py", "jet_transport.py")


class Cohort(str, Enum):
    faithful = "faithful"
    production = "production"


class RecheckPair(str, Enum):
    AB = "AB"
    AC = "AC"


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
            return f"anchor-{self.round:02d}-{self.scope.value}-{self.mode}-A"
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
            if round_index == 0:
                order = ("ABC", "ACB")[position % 2]
            else:
                order = ARM_ORDERS[(round_index + position) % len(ARM_ORDERS)]
            for arm in order:
                jobs.append(Job(round_index, scope, mode, arm))
    jobs.append(Job(REPETITIONS, Scope.combined, "full", "A", anchor=True))
    return jobs


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def faithful_snapshot_sources(result: Path, origin: Path) -> tuple[Path, dict[str, str]]:
    snapshot = result / "frozen-columnar"
    if snapshot.is_dir():
        manifest = dict(reversed(line.split(None, 1)) for line in
                        (snapshot / "source.sha256").read_text().splitlines())
        checked = {key: sha256(snapshot / key) for key in FAITHFUL_C_FILES}
        if checked != manifest:
            raise ValueError(f"Frozen faithful C source digest changed in {snapshot}")
        return snapshot, checked
    directory = origin.parent.parent
    if origin != directory / FAITHFUL_C_FILES[0]:
        raise ValueError("Faithful C source must include columnar/jets.py, columnar/inputs.py and jet_transport.py")
    return snapshot, {key: sha256(directory / key) for key in FAITHFUL_C_FILES}


def create_faithful_snapshot(snapshot: Path, original_script: Path, expected: dict[str, str]) -> None:
    if snapshot.is_dir():
        return
    temporary = snapshot.with_name(f".{snapshot.name}.{os.getpid()}")
    (temporary / "columnar").mkdir(parents=True)
    directory = original_script.parent.parent
    for key, digest in expected.items():
        shutil.copyfile(directory / key, temporary / key)
        if sha256(temporary / key) != digest:
            raise ValueError(f"Faithful source changed while freezing {key}")
    (temporary / "source.sha256").write_text("".join(f"{digest}  {key}\n" for key, digest in expected.items()))
    temporary.rename(snapshot)


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


def faithful_startup_probe(child: dict, family: list[dict], interpreter: Path) -> bool:
    # Exact command strings observed with the release ROOT library under GDB.
    sed_program = ("/^.include/,$" + "{", r"/^ \/.*++/p", "}")
    compiler = ("g++", "-xc++", "-E", "-v", "/dev/null")
    filter_command = ("sed", "-n", "-e", sed_program[0], "-e", sed_program[1], "-e", sed_program[2])
    quoted_filter = "sed -n " + " ".join(f"-e '{sentence}'" for sentence in sed_program)
    shell_commands = {
        "LD_DEBUG=libs LD_PRELOAD=DOESNOTEXIST ls 2>&1",
        "LC_ALL=C " + " ".join(compiler) + " 2>&1 | " + quoted_filter,
        f"ldd '{interpreter}'",
    }

    def known_command(row):
        args = tuple(row["argv"])
        if row["during_events"] or not args:
            return False
        tool = Path(args[0]).name
        if tool in {"sh", "bash"} and args[1:-1] in {("-c",), ("-c", "--")}:
            return " ".join(args[-1].split()) in shell_commands
        return ((tool, *args[1:]) in {compiler, filter_command, ("ldd", str(interpreter))}
                or args[1:] == ("/usr/bin/ldd", str(interpreter)) and tool in {"bash", "sh"})

    if known_command(child):
        return True
    # Only these leaves of an identified discovery subprocess may be sampled.
    allowed_leaves = {"cc1plus", "ls", "coreutils", "ld-linux-x86-64", "<exited>"}
    if child["comm"] not in allowed_leaves:
        return False
    return any(known_command(row) for row in family if row["pid"] in {child["parent_pid"], child["pid"]})


def verify_columnar(job: Job, attempt: Path, receipt: dict, inventory: dict,
                    source: dict, cohort: Cohort, core: Path, provider: Path | None,
                    wheel_members: dict, columnar_python: Path) -> dict:
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
    branch_count = re.findall(r"(?m)\), (\d+) input branches$", log)
    if len(branch_count) != 1:
        raise ValueError(f"Columnar branch count missing in {attempt}")
    branch_digest = re.findall(r"(?m)^Input branch SHA-256: ([0-9a-f]{64})$", log)
    items = {"input_branches": int(branch_count[0]), "input_branch_sha256": branch_digest[0] if branch_digest else None}
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
        site = (columnar_python.parent.parent / "lib/python3.14/site-packages").resolve(strict=True)
        package = site / "correctionlib"
        for name, digest in wheel_members["native"].items():
            if name.startswith("_core."):
                expected = package / name
                matches = {path for path in loaded if path.parent.name == "correctionlib"
                           and path.name.startswith("_core.") and path.name.endswith(".so")}
            else:
                expected = package / "lib" / name
                family = name.split(".so", 1)[0] + ".so"
                matches = {path for path in loaded if path.name == family or path.name.startswith(family + ".")}
            required = job.mode != "stored" or name in {"libstdc++.so.6", "libgcc_s.so.1"}
            if matches != ({expected} if required else set()) or (required and sha256(expected) != digest):
                raise ValueError(f"Production wheel runtime differs for {name} in {attempt}: {sorted(matches)}")
        data = site / "atlas_jet_demo/data"
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
    columnar_sources: dict[Path, str]
    c_provider: Path | None
    c_payload: Path | None
    wheel_members: dict
    matched_read: bool
    projections: dict[Scope, dict]
    projection_stamps: dict[Path, tuple]
    compact_defaults: frozenset[Scope]


def check_materialized_payload(attempt: Path, projection: dict | None) -> str:
    log = (attempt / "columnar.log").read_text()
    digests = re.findall(r"(?m)^Payload materialized SHA-256: ([0-9a-f]{64})$", log)
    counts = re.findall(r"(?m)^Payload materialized bytes: (\d+)$", log)
    if not projection:
        if digests or counts:
            raise ValueError(f"Unexpected scoped extraction in {attempt}")
        return "canonical or stored"
    if not (digests or counts) and not projection["runtime_identity_log_required"]:
        return "verified replay of the unchanged installed projector and input bytes"
    document = projection["document"]
    if digests != [document["sha256"]] or counts != [str(document["bytes"])]:
        raise ValueError(f"Timed C materialized different payload bytes in {attempt}")
    return "SHA-256 and byte count reported by the timed materialization"


def execute(job: Job, attempt: Path, context: Context, inventory: dict,
            specification: dict, diagnostic: bool) -> dict:
    output = attempt / "histograms.npz"
    projection = context.projections.get(job.scope) if job.mode != "stored" else None
    payload_receipt = None
    if projection and job.arm in {"B", "C"}:
        payload_receipt = {
            "scope": job.scope.value, "sha256": projection["document"]["sha256"],
            "bytes": projection["document"]["bytes"], "subtrees_sha256": projection["subtrees_sha256"],
        }
    if job.arm == "C":
        for path, digest in context.columnar_sources.items():
            if sha256(path) != digest:
                raise ValueError(f"Columnar source changed before {job.key}: {path}")
        scope = "both" if job.scope == Scope.combined else job.scope.value
        command = [
            str(context.columnar_python), str(context.columnar_script), str(context.input_file),
            str(output), "--mode", job.mode, "--scope", scope, "--repeat", "1", "--events", str(EVENTS),
        ]
        use_matched_read = context.matched_read and job.mode != "stored"
        if use_matched_read:
            command.append("--matched-read")
        if context.cohort == Cohort.faithful:
            command.extend(("--states", str(INVENTORY), "--histogram-spec", str(SPECIFICATION)))
            if job.mode != "stored":
                command.extend(("--provider", str(context.c_provider), "--payload", str(context.c_payload)))
        else:
            command.extend(compact_cli_flags(job.scope, job.mode, projection is not None, context.compact_defaults))
        if projection:
            for path in projection["shipped_component_sha256"]:
                path = Path(path)
                if file_stamp(path) != context.projection_stamps[path]:
                    raise ValueError(f"Shipped payload input changed before {job.key}: {path}")
        environment = dict(context.columnar_env)
        if diagnostic and context.cohort == Cohort.production:
            environment["ATLAS_JET_LIBRARY_AUDIT"] = str(attempt / "wheel-runtime.json")
        if context.cohort == Cohort.faithful and job.mode != "stored":
            interpreter = context.columnar_python.resolve()
            leg = run_leg(command, environment, attempt, "columnar", event_source=context.input_file,
                          startup_validator=lambda child, family: faithful_startup_probe(child, family, interpreter))
            if not leg["source_mapping_observed"]:
                raise ValueError(f"Could not verify the source mapping and processing interval in {attempt}")
        else:
            leg = run_leg(command, environment, attempt, "columnar",
                          capture_startup_libraries=context.cohort == Cohort.production)
        counters = verify_columnar(
            job, attempt, leg, inventory, context.source, context.cohort,
            context.core, context.c_provider, context.wheel_members, context.columnar_python,
        )
        counters["matched_read"] = use_matched_read
        if context.cohort == Cohort.production:
            reported = check_materialized_payload(attempt, projection)
            if projection:
                for path in projection["shipped_component_sha256"]:
                    path = Path(path)
                    if file_stamp(path) != context.projection_stamps[path]:
                        raise ValueError(f"Shipped payload input changed during {job.key}: {path}")
                payload_receipt["loader"] = "shipped on-demand extraction inside timed C"
                payload_receipt["identity_evidence"] = reported
            elif job.mode == "stored":
                payload_receipt = {"loader": "stored workflow without numerical assets"}
        if use_matched_read and not counters["input_branch_sha256"]:
            raise ValueError(f"Matched-read execution did not report actual input branch identity in {attempt}")
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
        bypass_overlay = context.cohort == Cohort.production and job.mode == "stored"
        environment = context.framework_env if job.arm == "A" or bypass_overlay else context.overlay_env
        if job.arm == "B" and projection:
            selected = Path(projection["document"]["path"])
            if file_stamp(selected) != context.projection_stamps[selected]:
                raise ValueError(f"Prebuilt scoped payload changed before {job.key}: {selected}")
            environment = {**environment, "JETS_PAYLOAD": str(selected)}
            payload_receipt.update(path=str(selected), loader="framework JETS_PAYLOAD")
            sealed_json(attempt / "framework-payload.json", payload_receipt)
        frame = run_leg(command, environment, attempt, "framework")
        checked_job = Job(job.round, job.scope, job.mode, "A", job.anchor) if bypass_overlay else job
        verify_framework(checked_job, attempt, frame, inventory, context.libraries, context.b_provider, context.core)
        if job.arm == "B" and projection and file_stamp(selected) != context.projection_stamps[selected]:
            raise ValueError(f"Prebuilt scoped payload changed during {job.key}: {selected}")
        if job.arm == "B" and bypass_overlay:
            if any(Path(x).name in {"libjets.so", "libcorrectionlib.so"} for x in frame["libraries"]):
                raise ValueError(f"Stored B loaded numerical assets in {attempt}")
            payload_receipt = {"loader": "stock stored workflow without numerical assets"}
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
        **({"payload_selection": payload_receipt} if payload_receipt else {}),
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
                    if job.arm == "C":
                        nominal_reads = jobs[job][1]["input_visit_counters"]
                        full_reads = jobs[Job(job.round, job.scope, "full", job.arm)][1]["input_visit_counters"]
                        if nominal_reads.get("matched_read"):
                            fields = ("input_branches", "input_branch_sha256")
                            if any(nominal_reads[key] != full_reads[key] for key in fields):
                                raise ValueError(f"Matched C inputs differ for {job.scope.value}, round {job.round}")
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


def recheck_schedule(scope: Scope, mode: str, pair: RecheckPair) -> list[Job]:
    jobs = [Job(0, scope, mode, "A", anchor=True)]
    for round_index in range(REPETITIONS):
        arms = pair.value if round_index % 2 == 0 else pair.value[::-1]
        jobs.extend(Job(round_index, scope, mode, arm) for arm in arms)
    jobs.append(Job(REPETITIONS, scope, mode, "A", anchor=True))
    return jobs


def recheck_measurement(pair: RecheckPair, mode: str, rows: dict[str, list[dict]], drift: dict) -> dict:
    ends = {"A": ("framework",), "B": ("framework",)} if pair == RecheckPair.AB else {
        "A": ("framework", "histogram", "total"), "C": ("total",),
    }
    measurements = {
        arm: {leg: check_quality([{"wall_s": row[leg]} for row in rows[arm]]) for leg in legs}
        for arm, legs in ends.items()
    }
    if pair == RecheckPair.AB:
        numerator, denominator, leg = "B", "A", "framework"
        limit, lower_pass = BUDGET, True
        name = "B_over_A_framework"
    else:
        numerator, denominator, leg = "A", "C", "total"
        limit, lower_pass = (10.0 if mode == "full" else None), False
        name = "A_over_C_end_to_end"
    left = measurements[numerator][leg]
    right = measurements[denominator][leg]
    ratio = left["mean_s"] / right["mean_s"]
    paired = [a / b for a, b in zip(left["retained"], right["retained"], strict=True)]
    quality = (all(value["pass"] for legs in measurements.values() for value in legs.values())
               and all(drift[leg]["pass"] for leg in ends["A"]))
    threshold_pass = None
    straddles = False
    if limit is not None:
        def passes(value):
            return value <= limit if lower_pass else value >= limit

        straddles = len({passes(value) for value in paired}) > 1
        if quality and not straddles:
            threshold_pass = passes(ratio)
    return {
        "name": name, "ratio_of_retained_means": ratio, "same_round_ratios": paired,
        "threshold": limit, "threshold_kind": "framework budget" if lower_pass else (
            "full-systematics aspiration" if limit is not None else None
        ),
        "threshold_pass": threshold_pass, "retained_pairs_straddle": straddles,
        "quality_pass": quality, "ratio_qualified": quality and not straddles,
        "quality": measurements,
    }


def paired_anchor_drift(anchors: list[dict], stock_rows: list[dict]) -> dict:
    # Compare both narrow stock brackets against the median stock observation
    # from the three retained pairs for this same scope and mode.
    result = {}
    for leg in anchors[0]:
        values = [row[leg] for row in anchors]
        median = statistics.median(row[leg] for row in stock_rows[1:])
        deviation = max(abs(value - median) for value in values) / median
        result[leg] = {"wall_s": values, "retained_stock_median_s": median,
                       "max_median_deviation": deviation, "pass": deviation <= MAX_DEVIATION_FROM_MEDIAN}
    return result


def recheck_summary(jobs: dict[Job, tuple], scheduled: list[Job], pair: RecheckPair) -> dict:
    summary = {
        "completed": len(jobs), "planned": len(scheduled),
        "progress": {job.key: str(attempt) for job, (attempt, _) in jobs.items()},
    }
    if len(jobs) != len(scheduled):
        return summary
    rows = {arm: [jobs[job][1]["wall_s"] for job in scheduled if not job.anchor and job.arm == arm]
            for arm in pair.value}
    anchors = [jobs[job][1]["wall_s"] for job in scheduled if job.anchor]
    drift = paired_anchor_drift(anchors, rows["A"])
    measured = recheck_measurement(pair, scheduled[0].mode, rows, drift)
    return {**summary, "stock_anchor_drift": drift, "comparison": measured,
            "quality_pass": measured["quality_pass"], "ratio_qualified": measured["ratio_qualified"]}


def compatible_journal(original: dict, provenance: dict, controller_sources: tuple[Path, ...]) -> bool:
    original = {**original, "identity_sha256": dict(original["identity_sha256"])}
    # Prior journals stored controller code among the physics input identities.
    # Their per-invocation receipts separately account for controller changes.
    for path in controller_sources:
        original["identity_sha256"].pop(str(path), None)
    return original == provenance


def original_recheck_authority(origin: Path, provenance: dict, controller_sources: tuple[Path, ...],
                               pair: RecheckPair, scope: Scope, mode: str) -> dict:
    campaign = origin / "campaign.json"
    summary = origin / "summary.json"
    initial = json.loads(campaign.read_text())
    if not compatible_journal(initial, provenance, controller_sources):
        raise ValueError("The requested artifacts and workload differ from the original production cohort")
    saved = json.loads(summary.read_text())
    if saved["completed"] != len(schedule()) or saved["planned"] != len(schedule()) or "ratios" not in saved:
        raise ValueError("A paired recheck requires the completed original production journal")

    historical = [Job(round_index, scope, mode, arm)
                  for round_index in range(REPETITIONS) for arm in pair.value]
    extra = {}
    if scope != Scope.combined:
        extra["combined_stock"] = Job(0, Scope.combined, mode, "A")
    if mode == "nominal":
        extra["full_stock"] = Job(0, scope, "full", "A")
        if pair == RecheckPair.AC and provenance["columnar_matched_read"]:
            extra["full_columnar"] = Job(0, scope, "full", "C")
    receipts = {}
    files = {}
    for job in (*historical, *extra.values()):
        accepted = selected_attempt(origin / job.key)
        if accepted is None:
            raise ValueError(f"Missing original accepted output for {job.key}")
        attempt, receipt = accepted
        if receipt["job"] != job.details() or str(attempt) != saved["progress"][job.key]:
            raise ValueError(f"Original receipt or journal pointer changed for {job.key}")
        if receipt["classification"] != ("warmup" if job.round == 0 else "retained"):
            raise ValueError(f"Original attempt classification differs for {job.key}")
        receipts[job] = receipt
        files[job] = {key: {"path": str(attempt / name), **identity(attempt / name)}
                      for key, name in (("histogram", "histograms.npz"), ("receipt", "receipt.json"))}

    stock = historical[0]
    def histogram(job):
        return Path(files[job]["histogram"]["path"])

    for job in historical[1:]:
        exact_archive(histogram(job), histogram(stock))
    if "combined_stock" in extra:
        exact_archive(histogram(stock), histogram(extra["combined_stock"]), projection=True)
    if "full_stock" in extra:
        check_pair(histogram(stock), histogram(extra["full_stock"]))

    rows = {arm: [receipts[job]["wall_s"] for job in historical if job.arm == arm] for arm in pair.value}
    prior = recheck_measurement(pair, mode, rows, saved["stock_anchor_drift"])
    if prior["ratio_qualified"]:
        raise ValueError("Original ratio passed the applicable spread gates and does not straddle its threshold")
    reason = []
    if not prior["quality_pass"]:
        reason.append("relevant original endpoint spread or stock drift did not qualify")
    if prior["retained_pairs_straddle"]:
        reason.append("original same-round pairs straddle the threshold")
    guards = {"stock": files[stock], "candidate": files[historical[1]],
              **{name: files[job] for name, job in extra.items()}}
    if pair == RecheckPair.AC:
        fields = ("matched_read", "input_branches", "input_branch_sha256")
        reported = receipts[historical[1]]["input_visit_counters"]
        guards["columnar_read"] = {key: reported[key] for key in fields}
        if "full_columnar" in extra:
            exact_archive(histogram(extra["full_columnar"]), histogram(extra["full_stock"]))
            full_reported = receipts[extra["full_columnar"]]["input_visit_counters"]
            if any(reported[key] != full_reported[key] for key in fields):
                raise ValueError("Original matched nominal/full C physical reads differ")
    return {
        "pair": pair.value, "scope": scope.value, "mode": mode, "reason": reason,
        "origin_campaign": {"path": str(campaign), **identity(campaign)},
        "origin_summary": {"path": str(summary), **identity(summary)},
        "original_ratio": prior, "guards": guards,
        "original_attempts": {job.key: files[job] for job in files},
    }


def check_recheck_job(job: Job, attempt: Path, receipt: dict, recheck: dict) -> None:
    output = attempt / "histograms.npz"
    guards = recheck["guards"]

    def checked_path(recorded, label):
        path = Path(recorded["path"])
        if identity(path) != {key: recorded[key] for key in ("sha256", "bytes")}:
            raise ValueError(f"Previously verified {label} changed: {path}")
        return path

    def checked_histogram(name):
        evidence = guards[name]
        checked_path(evidence["receipt"], f"{name} receipt")
        return checked_path(evidence["histogram"], f"{name} histogram")

    for label in ("origin_campaign", "origin_summary"):
        checked_path(recheck[label], label)
    exact_archive(output, checked_histogram("stock"))
    if job.arm != "A":
        exact_archive(output, checked_histogram("candidate"))
    if "combined_stock" in guards:
        exact_archive(output, checked_histogram("combined_stock"), projection=True)
    if "full_stock" in guards:
        check_pair(output, checked_histogram("full_stock"))
    if job.arm == "C":
        current = receipt["input_visit_counters"]
        if any(current[key] != value for key, value in guards["columnar_read"].items()):
            raise ValueError(f"Paired C physical input reads differ for {job.key}")


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


def identity(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def file_stamp(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def script_entrypoint(script: Path) -> ast.FunctionDef:
    entry, = (node for node in ast.parse(script.read_text()).body if isinstance(node, ast.FunctionDef)
              and node.name == "main")
    return entry


def script_defaults(entry: ast.FunctionDef) -> dict[str, tuple[ast.arg, ast.expr]]:
    arguments = entry.args.posonlyargs + entry.args.args
    positional = zip(arguments[len(arguments) - len(entry.args.defaults):], entry.args.defaults, strict=True)
    keywords = zip(entry.args.kwonlyargs, entry.args.kw_defaults, strict=True)
    return {arg.arg: (arg, value) for arg, value in (*positional, *keywords) if value is not None}


def script_bool_default(entry: ast.FunctionDef, name: str) -> bool | None:
    option = script_defaults(entry).get(name)
    if option is None:
        return None
    value = ast.literal_eval(option[1])
    if type(value) is not bool:
        raise ValueError(f"The shipped {name} default must be a boolean")
    return value


def same_ast(node: ast.AST, source: str, mode: str = "eval") -> bool:
    parsed = ast.parse(source, mode=mode)
    expected = parsed.body[0] if mode == "exec" else parsed.body
    return ast.dump(node) == ast.dump(expected)


def compact_contract(entry: ast.FunctionDef, bundle: dict) -> tuple[frozenset[Scope], bool, bool]:
    """Read actual default scopes, support, and a separate explicit off flag."""
    singles = frozenset({Scope.small, Scope.large})
    options = script_defaults(entry)
    option = options.get("compact_payload")
    advertised = bundle.get("demo_defaults")
    indexed = "scope_index" in bundle
    if indexed != (option is not None):
        raise ValueError("The visible compact option and sealed payload index disagree")

    nullable = option is not None and ast.literal_eval(option[1]) is None
    if nullable:
        # Match the generated guard and RHS rather than treating None as false.
        guards = [
            node for node in entry.body if isinstance(node, ast.If)
            and same_ast(node.test, "compact_payload is None")
        ]
        if len(guards) != 1 or len(guards[0].body) != 1 or guards[0].orelse:
            raise ValueError("The visible nullable compact option has no single default scope guard")
        assign = guards[0].body[0]
        if not isinstance(assign, ast.Assign) or not isinstance(assign.value, ast.BoolOp):
            raise ValueError("The visible compact default does not test the bundled payload and scope")
        scope_test = assign.value.values[-1]
        if not isinstance(scope_test, ast.Compare) or len(scope_test.comparators) != 1:
            raise ValueError("The visible compact default does not declare literal scopes")
        scope_values = ast.literal_eval(scope_test.comparators[0])
        if (type(scope_values) is not tuple or set(scope_values) - {item.value for item in singles}
                or len(set(scope_values)) != len(scope_values)
                or not same_ast(assign, f"compact_payload = packaged_payload and scope.value in {scope_values!r}", "exec")):
            raise ValueError("The visible compact default scope expression changed")
        declaration = option[0].annotation
        if declaration is None or not any(
            isinstance(node, ast.Constant) and node.value == "--compact-payload/--no-compact-payload"
            for node in ast.walk(declaration)
        ):
            raise ValueError("The visible compact option has no explicit on/off switch")
        actual = frozenset(Scope(scope) for scope in scope_values)
    else:
        actual = singles if script_bool_default(entry, "compact_payload") else frozenset()

    if advertised is not None:
        declared = advertised["compact_payload_default_scopes"]
        reuse_gate = ('reuse_nominal_nnjvt_update = reuse_nominal_nnjvt_update '
                      'and mode == Mode.full and "sr" in selected')
        if (type(declared) is not list or len(set(declared)) != len(declared)
                or set(declared) != {scope.value for scope in actual}
                or advertised["scope"] != "both" or advertised["mode"] != "full"
                or not same_ast(options["scope"][1], "Scope.both")
                or not same_ast(options["mode"][1], "Mode.full")
                or type(advertised["reuse_nominal_nnjvt_update"]) is not bool
                or script_bool_default(entry, "reuse_nominal_nnjvt_update")
                != advertised["reuse_nominal_nnjvt_update"]
                or advertised["reuse_nominal_nnjvt_update_modes"] != ["full"]
                or advertised["reuse_nominal_nnjvt_update_scopes"] != ["small", "both"]
                or sum(same_ast(node, reuse_gate, "exec") for node in entry.body) != 1):
            raise ValueError("The visible default configuration differs from the sealed bundle")
    elif nullable:
        raise ValueError("The visible nullable compact defaults require the sealed demo_defaults receipt")
    return actual, indexed, nullable


def signed_pair_contract(entry: ast.FunctionDef, bundle: dict) -> frozenset[Scope]:
    """Check the shipped internal gate against the sealed demo scope list."""
    setting = "reuse_uncertainty_sign_pairs"
    field = "reuse_uncertainty_sign_pair_scopes"
    defaults = bundle.get("demo_defaults") or {}
    configured = defaults.get(field, [])
    visible_scopes = {"small": Scope.small, "large": Scope.large, "both": Scope.combined}
    if (type(configured) is not list
            or any(type(scope) is not str or scope not in visible_scopes for scope in configured)
            or len(set(configured)) != len(configured)):
        raise ValueError("The sealed signed-state scopes must be a list of distinct small, large or both values")
    flags = {"--reuse-uncertainty-sign-pairs", "--no-reuse-uncertainty-sign-pairs"}
    for node in ast.walk(entry.args):
        if (isinstance(node, ast.arg) and node.arg == setting
                or isinstance(node, ast.Constant) and isinstance(node.value, str)
                and flags.intersection(node.value.split("/"))):
            raise ValueError("The shipped demo must keep signed-state reuse out of its CLI")

    ordered = tuple(scope for scope in visible_scopes if scope in configured)
    gate = f"{setting} = mode == Mode.full and scope.value in {ordered!r}"
    names = [node for node in ast.walk(entry) if isinstance(node, ast.Name) and node.id == setting]
    stores = [node for node in names if isinstance(node.ctx, (ast.Store, ast.Del))]
    allowed = (gate,) if field in defaults else (gate, f"{setting} = False")
    assignments = [node for node in entry.body if any(same_ast(node, line, "exec") for line in allowed)]
    forwarded = [node for node in ast.walk(entry) if isinstance(node, ast.keyword) and node.arg == setting]
    if field in defaults or names:
        if (len(stores) != 1 or len(assignments) != 1 or len(forwarded) != 1
                or not same_ast(forwarded[0].value, setting)):
            raise ValueError("The visible signed-state reuse gate differs from the sealed bundle")
    elif any(not same_ast(node.value, "False") for node in forwarded):
        raise ValueError("The visible script enables signed-state reuse without a sealed declaration")
    return frozenset(visible_scopes[scope] for scope in ordered)


def compact_cli_flags(scope: Scope, mode: str, projected: bool,
                      defaults: frozenset[Scope]) -> tuple[str, ...]:
    if scope == Scope.combined or mode == "stored":
        return ()
    if projected and scope not in defaults:
        return ("--compact-payload",)
    if not projected and scope in defaults:
        return ("--no-compact-payload",)
    return ()


def projection_node_hashes(canonical: Path, layout: dict, manifest: dict, scope: Scope,
                           correction_names: list[str]) -> dict:
    spans = layout["scopes"][scope.value]["spans"]
    if correction_names != [name for name in manifest["canonical_order"] if name in correction_names]:
        raise ValueError(f"Noncanonical correction order for {scope.value}")
    hashes = {}
    with canonical.open("rb") as source:
        for name, (start, stop) in zip(correction_names, spans, strict=True):
            entry = manifest["nodes"][name]
            if stop - start != entry["bytes"]:
                raise ValueError(f"Index byte dimensions differ for {name}")
            digest = hashlib.sha256()
            source.seek(start)
            remaining = stop - start
            while remaining:
                block = source.read(min(remaining, 1024 * 1024))
                if not block:
                    raise ValueError(f"Canonical document ended inside {name}")
                digest.update(block)
                remaining -= len(block)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"Canonical raw correction bytes differ for {name}")
            hashes[name] = digest.hexdigest()
    return hashes


def compare_scoped_ntuple(stock: Path, candidate: Path, events: int) -> dict:
    """Compare every public persisted branch of the isolated stock and B endpoints."""
    from compare_observer import difference

    proofs = {}
    mismatches = {}
    with open_root(stock / "reference.root") as a_file, open_root(candidate / "reference.root") as b_file:
        a, b = a_file["analysis"], b_file["analysis"]
        if a.num_entries != events or b.num_entries != events:
            raise ValueError("Scoped stock and B ntuples do not have the requested entry interval")
        keys = sorted(a.keys())
        if keys != sorted(b.keys()) or not keys:
            raise ValueError(f"Scoped stock/B persisted keys differ: {sorted(set(keys) ^ set(b.keys()))[:12]}")
        # Production trees have no explicit event/jet ID fields; their vector
        # lengths and payload are checked in the stored event-entry order.
        unused_identity = np.zeros(events, dtype=np.int64)
        for start in range(0, len(keys), 24):
            packet = keys[start:start + 24]
            left, right = a.arrays(packet, how=dict), b.arrays(packet, how=dict)
            for key in packet:
                if a[key].typename != b[key].typename:
                    raise ValueError(f"Scoped persisted ROOT branch type differs for {key}")
                delta, proof = difference(key, left[key], right[key], unused_identity, unused_identity)
                proofs[key] = proof
                if delta is not None:
                    for example in delta["examples"]:
                        example.pop("run")
                        example.pop("event")
                    mismatches[key] = delta
    with (open_root(stock / "hist-reference.root") as a_file,
          open_root(candidate / "hist-reference.root") as b_file):
        labels = {}
        for key in ("systematicsList", "systematicsListOnlyWeights"):
            left = a_file[key].axis().labels() if key in a_file else None
            right = b_file[key].axis().labels() if key in b_file else None
            if left != right:
                raise ValueError(f"Scoped systematic order differs in {key}")
            labels[key] = left
    return {
        "stock_ntuple": {"path": str(stock / "reference.root"), **identity(stock / "reference.root")},
        "framework_ntuple": {"path": str(candidate / "reference.root"), **identity(candidate / "reference.root")},
        "stock_counters": {"path": str(stock / "hist-reference.root"), **identity(stock / "hist-reference.root")},
        "framework_counters": {"path": str(candidate / "hist-reference.root"), **identity(candidate / "hist-reference.root")},
        "events": events, "keys": keys, "branches": len(keys), "systematic_labels": labels,
        "checked_values": sum(row["items"] for row in proofs.values()),
        "checked_value_bytes": sum(row["bytes"] for row in proofs.values()),
        "exact": not mismatches, "differences": mismatches, "proofs": proofs,
    }


def scoped_framework_proof(path: Path, scope: Scope, selected: Path, source: Path, provider: Path,
                           core: Path, libraries: dict, inventory: dict) -> dict:
    recovery = json.loads(path.read_text())
    record_path = Path(recovery["pilot_provenance"]["path"]).resolve(strict=True)
    if recovery["pilot_provenance"]["sha256"] != sha256(record_path):
        raise ValueError(f"Original B pilot provenance differs from its exact endpoint receipt: {path}")
    record = json.loads(record_path.read_text())
    pilot = record_path.parent
    events = record["events"]
    if (record["path"] != "B" or record["scope_collections"] != scope.value
            or record["histogram_spec_sha256"] != sha256(SPECIFICATION)
            or events != EVENTS or Path(record["input"]).resolve() != source.resolve()
            or not recovery["endpoint_exact"] or recovery["process_audit_reclassified"]
            or recovery["scope"] != scope.value or recovery["events"] != events
            or Path(recovery["input"]).resolve() != source.resolve()):
        raise ValueError(f"Scoped B receipt does not cover {scope.value} and the frozen input: {path}")
    expected_artifacts = (selected, provider, core, *libraries.values())
    for artifact in expected_artifacts:
        if record["artifact_sha256"].get(str(artifact)) != sha256(artifact):
            raise ValueError(f"Scoped B receipt does not identify {artifact}")
    stock = Path(record["stock"]).resolve(strict=True)
    stock_record = json.loads((stock / "provenance.json").read_text())
    if (stock_record["events"] != events or stock_record["scope_collections"] != scope.value
            or Path(stock_record["input"]).resolve() != source.resolve()):
        raise ValueError(f"Scoped B stock receipt differs from the selected interval: {stock}")
    exact = {}
    for mode in MODES:
        candidate = pilot / f"01-{mode}"
        frame_path = candidate / "framework.json"
        checked = recovery["modes"][mode]
        if checked["framework_receipt"] != {"path": str(frame_path), **identity(frame_path)}:
            raise ValueError(f"Scoped B process receipt differs for {scope.value}/{mode}")
        frame = json.loads(frame_path.read_text())
        if frame["exit"] != 0 or frame["threads_seen"] != [1] or frame["cpus_seen"] != [str(CPU)]:
            raise ValueError(f"Scoped B execution proof is incomplete for {scope.value}/{mode}")
        confirm_scope(candidate, mode, scope, inventory, events)
        if mode != "stored":
            for library in (provider, core, libraries["JetUncertaintiesLib"], libraries["JetsCorrectionRuntime"]):
                verified_library(frame, library, True)
        actual = Path(checked["endpoint"]["path"])
        expected = stock / f"01-{mode}/histograms.npz"
        if checked["endpoint"] != {"path": str(actual), **identity(actual)} or checked["stock_endpoint"] != {
            "path": str(expected), **identity(expected),
        }:
            raise ValueError(f"Scoped B histogram input differs for {scope.value}/{mode}")
        exact_archive(actual, expected)
        exact[mode] = sha256(actual)
    comparison_file = Path(recovery["full_ntuple_comparison"]["path"])
    if recovery["full_ntuple_comparison"] != {"path": str(comparison_file), **identity(comparison_file)}:
        raise ValueError(f"Scoped B all-branch comparison moved since acceptance: {comparison_file}")
    comparison = json.loads(comparison_file.read_text())
    if not comparison["exact"] or comparison["events"] != events or not comparison["branches"]:
        raise ValueError(f"Scoped B is missing its exact full persisted-branch comparison: {comparison_file}")
    for role, directory in (("stock", stock), ("framework", pilot)):
        for record_key, basename in (("ntuple", "reference.root"), ("counters", "hist-reference.root")):
            observed = directory / "01-full" / basename
            if comparison[f"{role}_{record_key}"] != {"path": str(observed), **identity(observed)}:
                raise ValueError(f"Scoped B {role} evidence changed: {observed}")
    return {"path": str(path), "sha256": sha256(path), "events": events, "histograms": exact,
            "full_branches": comparison["branches"], "full_values": comparison["checked_values"]}


def scoped_columnar_proof(path: Path, shipped: Path, scope: Scope, selected: Path,
                          authority: Path, source: Path, source_uuid: str, verified_provider: set,
                          verified_core: set, wheel: Path, script: Path, *,
                          reuse_uncertainty_sign_pairs: bool = False) -> dict:
    proof = json.loads(path.read_text())
    check_signed_pair_proof(proof, path, reuse_uncertainty_sign_pairs)
    if (not proof["passed"] or proof["scope"] != scope.value or proof["mode"] != "full"
            or proof["stage_limit"] != "final" or proof["entries_compared"] != EVENTS
            or Path(proof["authority"]).resolve() != authority
            or Path(proof["source_phySLITE"]).resolve() != source.resolve() or proof["source_file_uuid"] != source_uuid
            or sha256(Path(proof["payload"])) != sha256(selected)
            or sha256(Path(proof["provider"])) not in verified_provider
            or sha256(Path(proof["runtime"]["core"])) not in verified_core
            or proof.get("reuse_nominal_nnjvt_update", False)
            != bool(script_bool_default(script_entrypoint(script), "reuse_nominal_nnjvt_update"))):
        raise ValueError(f"Scoped C object receipt does not cover the actual projected bytes: {path}")
    check = json.loads(shipped.read_text())
    installation = check["installation"]
    if (check["scope"] != scope.value or Path(check["original"]).resolve() != source.resolve()
            or check["events"] != EVENTS or not check["all_histogram_fields_exact"]
            or not check["matched_nominal_nosys_exact"]
            or installation["wheel_sha256"] != sha256(wheel) or installation["script_sha256"] != sha256(script)):
        raise ValueError(f"Shipped scoped C acceptance differs from the timed artifact: {shipped}")
    stock = Path(check["stock_full_histogram"]["path"])
    if sha256(stock) != check["stock_full_histogram"]["sha256"]:
        raise ValueError(f"Stock histogram moved since shipped C acceptance: {stock}")
    with np.load(stock, allow_pickle=False) as archive:
        if archive["events"].item() != EVENTS:
            raise ValueError(f"Scoped shipped C histogram does not cover the complete input: {stock}")
    for name in ("pip-indexed-full", "pip-indexed-nominal", "uv-indexed-full"):
        if (shipped.parent / f"{name}.exit").read_text().strip() != "0":
            raise ValueError(f"Shipped scoped C did not complete {name}")
        command = json.loads((shipped.parent / f"{name}.argv.json").read_text())
        if "--compact-payload" not in command or "--payload" in command:
            raise ValueError(f"Shipped C test did not exercise on-demand projection: {name}")
    exact_archive(shipped.parent / "pip-indexed-full.npz", stock)
    exact_archive(shipped.parent / "uv-indexed-full.npz", stock)
    check_pair(shipped.parent / "pip-indexed-nominal.npz", stock)
    return {"objects": {"path": str(path), "sha256": sha256(path), "events": proof["entries_compared"]},
            "shipped": {"path": str(shipped), "sha256": sha256(shipped), "events": check["events"]}}


def verified_projections(partition: Path, acceptance: Path, scopes: set[Scope], canonical: Path,
                         wheel: Path, bundle: dict, script: Path, interpreter: Path, source: Path,
                         source_uuid: str, authority: Path, provider: Path, core: Path,
                         libraries: dict, wheel_members: dict, inventory: dict, *,
                         signed_pair_scopes: frozenset[Scope] = frozenset()) -> dict[Scope, dict]:
    manifest_path = partition / "manifest.json"
    index = partition / "payload-index.json"
    manifest = json.loads(manifest_path.read_text())
    layout = json.loads(index.read_text())
    combined = identity(canonical)
    if (not manifest["raw_node_identity_verified"] or layout.get("format") != 1
            or layout["canonical"] != combined or any(manifest["canonical"][k] != v for k, v in combined.items())
            or bundle.get("scope_index") != identity(index)):
        raise ValueError("The partition and shipped index do not describe the verified combined payload")

    directory = Path(subprocess.check_output(
        [str(interpreter), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        env=clean_environment(), text=True,
    ).strip()) / "atlas_jet_demo"
    installed = directory / "data"
    projector = directory / "payload_ranges.py"
    installed_members = {"atlas_jet_demo/data/payload-index.json": installed / "payload-index.json",
                         "atlas_jet_demo/payload_ranges.py": projector}
    with zipfile.ZipFile(wheel) as archive:
        for name, path in installed_members.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != sha256(path):
                raise ValueError(f"Installed compact member differs from the wheel: {path}")
    revision = bundle["git_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("The shipped projector needs an exact committed source revision")
    committed = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{revision}:exporters/payload_ranges.py"]
    )
    if (identity(projector) != bundle.get("scope_projector") or committed != projector.read_bytes()
            or identity(installed / "payload-index.json") != identity(index)
            or identity(installed / "jets.json") != combined):
        raise ValueError("The accepted canonical payload, index, or committed projector differs from production")
    project = runpy.run_path(str(projector))["materialize"]
    supplied = json.loads(acceptance.read_text())
    if set(supplied) != {scope.value for scope in scopes}:
        raise ValueError("Provide B, direct C and shipped C proofs for every selected scope")
    results = {}
    for scope in sorted(scopes, key=lambda x: x.value):
        entry = supplied[scope.value]
        relative = manifest["native_load_paths"][scope.value]
        target = (partition / relative).resolve(strict=True)
        if not target.is_relative_to(partition):
            raise ValueError(f"Scoped B payload escapes its sealed partition: {target}")
        declared = manifest["artifacts"][relative]
        actual = identity(target)
        expected = layout["scopes"][scope.value]
        if any(declared[key] != value or expected[key] != value for key, value in actual.items()):
            raise ValueError(f"Actual scoped document differs from the index and manifest: {target}")
        node_hashes = projection_node_hashes(canonical, layout, manifest, scope, declared["corrections"])
        with TemporaryDirectory(prefix="atlas-jet-verified-") as scratch:
            reconstructed = Path(scratch) / "jets.json"
            project(installed / "jets.json", installed / "payload-index.json", scope.value, reconstructed)
            if identity(reconstructed) != actual:
                raise ValueError(f"Verified shipped code reconstructed different bytes for {scope.value}")
        evidence = {name: (acceptance.parent / entry[name]).resolve(strict=True)
                    for name in ("b_recovery", "c_objects", "c_shipped")}
        results[scope] = {
            "document": {"path": str(target), **actual},
            "subtrees_sha256": node_hashes,
            "canonical": combined,
            "partition_manifest": {"path": str(manifest_path), **identity(manifest_path)},
            "index": {"path": str(index), **identity(index)},
            "shipped_component_sha256": {str(path): sha256(path) for path in (*installed_members.values(), installed / "jets.json")},
            "runtime_identity_log_required": "Payload materialized SHA-256:" in script.read_text(),
            "B": scoped_framework_proof(evidence["b_recovery"], scope, target, source, provider, core, libraries, inventory),
            "C": scoped_columnar_proof(
                evidence["c_objects"], evidence["c_shipped"], scope, target, authority, source, source_uuid,
                {sha256(provider), wheel_members["native"]["libjets.so"]},
                {sha256(core), wheel_members["native"]["libcorrectionlib.so"]}, wheel, script,
                reuse_uncertainty_sign_pairs=scope in signed_pair_scopes,
            ),
        }
    return results


def check_signed_pair_proof(proof: dict, path: Path, enabled: bool) -> None:
    if proof.get("reuse_uncertainty_sign_pairs", False) is not enabled:
        raise ValueError(f"Direct PHYSLITE verification used a different signed-state reuse setting: {path}")


def verify_acceptance(b: Path | None, c: Path | None, diagnostic: bool, overlay: Path,
                      provider: Path, payload: Path, core: Path, *,
                      production_input: tuple[Path, str, bool] | None = None,
                      reuse_uncertainty_sign_pairs: bool = False) -> dict:
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
        check_signed_pair_proof(proof, c, reuse_uncertainty_sign_pairs)
        if (proof["entries_compared"] != EVENTS or proof["scope"] != "both" or proof["mode"] != "full"
                or proof["stage_limit"] != "final" or not proof["passed"]):
            raise ValueError(f"Incomplete or failing direct PHYSLITE verification in {c}")
        if production_input is not None:
            original, uuid, reuse_nnjvt = production_input
            checked_input = proof.get("source_phySLITE")
            if (not isinstance(checked_input, str) or Path(checked_input).resolve() != original.resolve()
                    or proof.get("source_file_uuid") != uuid):
                raise ValueError(f"Direct PHYSLITE verification used a different original input: {c}")
            if proof.get("reuse_nominal_nnjvt_update", False) is not reuse_nnjvt:
                raise ValueError(f"Direct PHYSLITE verification used a different NNJVT reuse setting: {c}")
        proofs["C"] = {"path": str(c), "sha256": sha256(c), "authority": proof["authority"],
                       "provider": proof["provider"], "payload": proof["payload"],
                       "core": proof["runtime"]["core"]}
    return proofs


@APP.command()
def recover_b_pilot(pilot: Path, result: Path) -> None:
    """Reconstruct the histogram endpoint from a saved, successful B job."""
    pilot = pilot.resolve(strict=True)
    target = result.resolve()
    provenance = pilot / "provenance.json"
    record = json.loads(provenance.read_text())
    if record["path"] != "B" or not record["pilot"]:
        raise ValueError("Recovery requires an existing Path B pilot")
    scope = Scope(record["scope_collections"])
    inventory = json.loads(INVENTORY.read_text())
    events = record["events"]
    stock = Path(record["stock"]).resolve(strict=True)
    stock_record = json.loads((stock / "provenance.json").read_text())
    if (stock_record["events"], stock_record["scope_collections"], stock_record["input"]) != (
        events, scope.value, record["input"],
    ):
        raise ValueError("Stock and B pilots do not share the same scope and event interval")
    frames = {}
    for mode in MODES:
        candidate = pilot / f"01-{mode}"
        frame = candidate / "framework.json"
        original = json.loads(frame.read_text())
        if original["exit"] != 0:
            raise ValueError(f"Failed framework process output in {candidate}")
        confirm_scope(candidate, mode, scope, inventory, events)
        frames[mode] = {
            "framework_receipt": {"path": str(frame), **identity(frame)},
            "framework_log": {"path": str(candidate / "framework.log"), **identity(candidate / "framework.log")},
            "ntuple": {"path": str(candidate / "reference.root"), **identity(candidate / "reference.root")},
            "systematic_counters": {"path": str(candidate / "hist-reference.root"), **identity(candidate / "hist-reference.root")},
            "recorded_descendants": original["descendant_observations"],
            "recorded_threads": original["threads_seen"],
            "recorded_cpu": original["cpus_seen"],
        }
    target.mkdir()
    ntuple_path = target / "full-ntuple.json"
    ntuple = compare_scoped_ntuple(stock / "01-full", pilot / "01-full", events)
    sealed_json(ntuple_path, ntuple)
    if not ntuple["exact"]:
        raise ValueError(f"Scoped B persisted branches differ from stock: {ntuple_path}")
    for mode in MODES:
        candidate = pilot / f"01-{mode}"
        recovered = target / f"{mode}.npz"
        if (candidate / "histograms.npz").is_file():
            shutil.copyfile(candidate / "histograms.npz", recovered)
        else:
            subprocess.run([
                str(HISTOGRAM_PYTHON), str(REFERENCE / "workflow.py"), "histogram",
                str(candidate / "reference.root"), str(INVENTORY), str(SPECIFICATION),
                str(recovered), mode, "--scope", scope.value,
            ], env=clean_environment(), check=True)
        reference = stock / f"01-{mode}/histograms.npz"
        exact_archive(recovered, reference)
        frames[mode]["endpoint"] = {"path": str(recovered), **identity(recovered)}
        frames[mode]["stock_endpoint"] = {"path": str(reference), **identity(reference)}
    check_pair(target / "nominal.npz", target / "full.npz")
    sealed_json(target / "recovery.json", {
        "pilot_provenance": {"path": str(provenance), **identity(provenance)},
        "scope": scope.value, "events": events, "input": record["input"], "endpoint_exact": True,
        "classification": "read-only physics recovery, excluded from timing",
        "process_audit_reclassified": False, "modes": frames,
        "full_ntuple_comparison": {"path": str(ntuple_path), **identity(ntuple_path)},
    })


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
    matched_read: Annotated[bool, typer.Option(help="Match nominal/full columnar branch requests")] = False,
    compact_scope: Annotated[list[Scope] | None, typer.Option(help="Project small or large production payloads")] = None,
    payload_partition: Annotated[Path | None, typer.Option(help="Verified canonical partition and byte index")] = None,
    scoped_acceptance: Annotated[Path | None, typer.Option(help="JSON with b_recovery, c_objects and c_shipped per scope")] = None,
    diagnostic: Annotated[bool, typer.Option(help="Record visits without including them in timed cohorts")] = False,
    recheck_of: Annotated[Path | None, typer.Option(help="Completed production journal for a fresh paired recheck")] = None,
    recheck_scope: Annotated[Scope | None, typer.Option()] = None,
    recheck_mode: Annotated[str | None, typer.Option(help="One mode from the original journal")] = None,
    recheck_pair: Annotated[RecheckPair | None, typer.Option(help="AB framework budget or AC saved histogram")] = None,
) -> None:
    """Run or resume one exact full-file cohort on the reserved CPU."""
    if os.sched_getaffinity(0) != {CPU} or len(list(Path("/proc/self/task").iterdir())) != 1:
        raise typer.BadParameter(f"Launch the controller with one thread on CPU {CPU}")
    if diagnostic and jobs is None:
        raise typer.BadParameter("Set a job limit for a diagnostic invocation")

    result = result.resolve()
    requested_recheck = (recheck_of, recheck_scope, recheck_mode, recheck_pair)
    if any(requested_recheck) and not all(requested_recheck):
        raise typer.BadParameter("Provide --recheck-of, --recheck-scope, --recheck-mode and --recheck-pair together")
    if recheck_of:
        if cohort != Cohort.production or diagnostic or jobs is not None:
            raise typer.BadParameter("A fresh paired recheck requires --cohort production without --diagnostic or --jobs")
        if recheck_mode not in MODES or (recheck_pair == RecheckPair.AB and recheck_mode == "stored"):
            raise typer.BadParameter("Select nominal or full for the framework budget, or any original AC mode")
        recheck_of = recheck_of.resolve(strict=True)
        if result.exists() or result.is_relative_to(recheck_of) or recheck_of.is_relative_to(result):
            raise typer.BadParameter("Provide a new paired result directory outside the original journal")
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
    snapshot = None
    if cohort == Cohort.faithful:
        snapshot, source_hashes = faithful_snapshot_sources(result, columnar_script)
        physical_script = snapshot / FAITHFUL_C_FILES[0]
        columnar_sources = {snapshot / key: digest for key, digest in source_hashes.items()}
        entrypoint_sha256 = source_hashes[FAITHFUL_C_FILES[0]]
    else:
        physical_script = columnar_script
        entrypoint_sha256 = sha256(physical_script)
        columnar_sources = {physical_script: entrypoint_sha256}
    scopes = set(compact_scope or ())
    if Scope.combined in scopes:
        raise typer.BadParameter("Combined calibration uses the canonical document")
    if scopes and cohort != Cohort.production:
        raise typer.BadParameter("Scoped payload production requires the shipped wheel")
    wheel_proof = None
    compact_defaults = frozenset()
    signed_pair_scopes = frozenset()
    reuse_nnjvt = False
    if wheel:
        wheel_proof = wheel.with_name("bundle.json")
        bundle = json.loads(wheel_proof.read_text())
        if (wheel.parent != columnar_script.parent or bundle["wheel"]["sha256"] != sha256(wheel)
                or bundle["visible_script"]["sha256"] != sha256(columnar_script)
                or bundle["provider"]["sha256"] != sha256(provider)
                or bundle["payload"]["sha256"] != sha256(payload)
                or wheel_members["data"]["jets.json"] != sha256(payload)):
            raise ValueError("Production framework, visible script, or wheel differs from the bundle receipt")
        with zipfile.ZipFile(wheel) as archive:
            build = json.loads(archive.read("atlas_jet_demo/data/build.json"))
        if (build.get("demo_defaults") != bundle.get("demo_defaults")
                or build.get("scope_index") != bundle.get("scope_index")
                or ("visible_source" in build and build["visible_source"] != identity(physical_script))):
            raise ValueError("Production wheel and bundle default or visible-source receipts differ")
        entry = script_entrypoint(physical_script)
        compact_defaults, supports_compact, supports_override = compact_contract(entry, bundle)
        signed_pair_scopes = signed_pair_contract(entry, bundle)
        reuse_nnjvt = bool(script_bool_default(entry, "reuse_nominal_nnjvt_update"))
        if scopes and not supports_compact:
            raise typer.BadParameter("The shipped script does not expose compact payload loading")
        scopes = scopes or set(compact_defaults)
        if not supports_override and compact_defaults - scopes:
            raise typer.BadParameter("This earlier shipped script cannot turn off a default projection")
    if bool(scopes) != bool(payload_partition) or bool(scopes) != bool(scoped_acceptance):
        raise typer.BadParameter("Selected projections require a partition and per-scope acceptance manifest")
    if cohort == Cohort.faithful and (sha256(provider) != sha256(columnar_provider)
                                     or sha256(payload) != sha256(columnar_payload)):
        raise ValueError("Faithful paths B and C must use the same native provider and payload")

    source = json.loads(source_verification.read_text())
    input_spec = json.loads((ROOT / "config/jets-input.json").read_text())
    input_file = Path(input_spec["path"])

    production_input = (input_file, source["source_file_uuid"], reuse_nnjvt) if wheel else None
    proofs = verify_acceptance(
        b_verification, c_verification, diagnostic, overlay, provider, payload, core,
        production_input=production_input, reuse_uncertainty_sign_pairs=Scope.combined in signed_pair_scopes,
    )
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
        columnar_env = dict(framework_env)
        for key in ("PYTHONHOME", "PYTHONPATH"):
            columnar_env.pop(key, None)
        columnar_env["LD_LIBRARY_PATH"] = f"{sdk / 'lib'}:{framework_env['LD_LIBRARY_PATH']}"

    inventory = json.loads(INVENTORY.read_text())
    specification = json.loads(SPECIFICATION.read_text())
    projections = {}
    projection_stamps = {}
    if scopes:
        projections = verified_projections(
            payload_partition.resolve(strict=True), scoped_acceptance.resolve(strict=True), scopes,
            payload, wheel, bundle, physical_script, columnar_python, input_file, source["source_file_uuid"],
            independent_stock, provider, core, libraries, wheel_members, inventory,
            signed_pair_scopes=signed_pair_scopes,
        )
        for projection in projections.values():
            selected = projection["document"]
            verified_paths = {selected["path"]: selected["sha256"], **projection["shipped_component_sha256"]}
            for name, digest in verified_paths.items():
                path = Path(name)
                stamp = file_stamp(path)
                if sha256(path) != digest or file_stamp(path) != stamp:
                    raise ValueError(f"Scoped payload input changed during preflight: {path}")
                projection_stamps[path] = stamp
    controller_sources = (
        ROOT / "tools/benchmark_campaign.py", REFERENCE / "benchmark.py", ROOT / "framework/bench_path_b.py",
    )
    identity_paths = [
        REFERENCE / "workflow.py", REFERENCE / "setup.sh", INVENTORY, SPECIFICATION,
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
        "identity_sha256": {str(path): (entrypoint_sha256 if path == columnar_script else sha256(path))
                            for path in identity_paths},
        "wheel_members_sha256": wheel_members, "columnar_matched_read": matched_read,
        "endpoint": {"A": ["CPRun", "histogram"], "B": ["CPRun", "histogram"],
                     "C": ["one direct PHYSLITE histogram pass"]},
        "endpoint_clock": "monotonic wall including startup and output serialization",
        "schedule": [job.details() for job in scheduled],
    }
    if cohort == Cohort.production:
        provenance["columnar_startup_library_observations"] = STARTUP_LIBRARY_OBSERVATIONS
    if projections:
        provenance["projected_payloads"] = {scope.value: receipt for scope, receipt in projections.items()}
        # Keep the schema of earlier one-switch production journals.
        if bundle.get("demo_defaults") is None:
            provenance["compact_by_default"] = bool(compact_defaults)
        else:
            provenance["compact_default_scopes"] = sorted(scope.value for scope in compact_defaults)
        provenance["scoped_acceptance_sha256"] = sha256(scoped_acceptance)
    context = Context(
        cohort, input_file, source, framework_python, framework_env, overlay_env, columnar_env,
        libraries, provider, core, columnar_python, physical_script, columnar_sources,
        columnar_provider, columnar_payload, wheel_members, matched_read, projections, projection_stamps,
        compact_defaults,
    )

    with (ROOT / f"results/.benchmark-cpu{CPU}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recheck = None
        if recheck_of:
            require_no_live_attempt(recheck_of)
            recheck = original_recheck_authority(
                recheck_of, provenance, controller_sources, recheck_pair, recheck_scope, recheck_mode,
            )
            scheduled = recheck_schedule(recheck_scope, recheck_mode, recheck_pair)
            provenance = {**provenance, "schedule": [job.details() for job in scheduled], "timing_recheck": recheck}
        result.mkdir(parents=True, exist_ok=recheck is None)
        require_no_live_attempt(result)
        campaign = result / "campaign.json"
        if campaign.is_file():
            original = json.loads(campaign.read_text())
            if not compatible_journal(original, provenance, controller_sources):
                raise ValueError(f"Configuration or artifact identity changed for {result}")
        else:
            sealed_json(campaign, provenance)

        def summarize(journal):
            if recheck:
                return recheck_summary(journal, scheduled, recheck_pair)
            return campaign_summary(journal)

        if snapshot:
            create_faithful_snapshot(snapshot, columnar_script, source_hashes)
        controllers = result / "controllers"
        controllers.mkdir(exist_ok=True)
        controller = controllers / f"{time.time_ns()}-{os.getpid()}.json"
        sealed_json(controller, {
            "source_sha256": {str(path): sha256(path) for path in controller_sources},
            "columnar_source_sha256": {str(path): digest for path, digest in columnar_sources.items()},
            "columnar_entrypoint": str(physical_script),
            "columnar_runtime": {key: columnar_env[key] for key in
                                 ("PATH", "ROOTSYS", "ROOT_INCLUDE_PATH", "LD_LIBRARY_PATH") if key in columnar_env},
        })
        finished = {
            job: accepted for job in scheduled
            if (accepted := selected_attempt(result / job.key)) is not None
        }
        sealed_json(result / "summary.json", summarize(finished))
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
            sealed_json(attempt / "job.json", {**job.details(), "launch": launch, "controller": str(controller)})
            try:
                receipt = execute(job, attempt, context, inventory, specification, diagnostic)
            except Exception as error:
                sealed_json(attempt / "receipt.json", {
                    "job": job.details(), "classification": "diagnostic", "failure": str(error),
                    "launch": launch, "completed_unix_ns": time.time_ns(),
                })
                raise
            receipt.update(launch=launch, completed_unix_ns=time.time_ns())
            if not diagnostic:
                proposed = {**finished, job: (attempt, receipt)}
                try:
                    if recheck:
                        check_recheck_job(job, attempt, receipt, recheck)
                    summary = summarize(proposed)
                except (ValueError, AssertionError) as error:
                    receipt.update(classification="diagnostic", guard_failure=str(error))
                    sealed_json(attempt / "receipt.json", receipt)
                    raise
                finished = proposed
                sealed_json(attempt / "receipt.json", receipt)
                sealed_json(result / "summary.json", summary)
            else:
                sealed_json(attempt / "receipt.json", receipt)
            typer.echo(f"  {receipt['wall_s']['total']:.3f} s ({receipt['classification']})", err=True)
        summary = summarize(finished)
        typer.echo(f"{summary['completed']} of {summary['planned']} jobs accepted in {result}")
        if summary.get("quality_pass") is False or summary.get("ratio_qualified") is False:
            raise typer.Exit(2)


if __name__ == "__main__":
    APP()
