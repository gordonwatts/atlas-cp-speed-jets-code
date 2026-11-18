"""Time stock framework jobs and the matched separate histogram leg."""

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import typer

from workflow import Scope, branch_states, open_root

APP = typer.Typer(add_completion=False)
ROOT = Path(__file__).resolve().parents[2]
MODES = ("stored", "nominal", "full")
EVENTS = 20000
CPU = 3
REPETITIONS = 4
MAX_COEFFICIENT_OF_VARIATION = 0.05
MAX_DEVIATION_FROM_MEDIAN = 0.10
ROOT_DISCOVERY_COMMANDS = {"bash", "sh", "ls", "coreutils", "g++", "cc1plus", "sed", "ldd"}


def release_environment() -> dict[str, str]:
    completed = subprocess.run(
        ["bash", "-c", 'source "$1" >&2 && env -0', "_",
         str(ROOT / "tools/reference/setup.sh")], capture_output=True, check=True
    )
    return dict(item.split("=", 1) for item in os.fsdecode(completed.stdout).rstrip("\0").split("\0"))


def sample_activity() -> dict:
    with open("/proc/stat") as stream:
        counters = {row[0]: [int(value) for value in row[1:]]
                    for line in stream if (row := line.split())[0] in ("cpu", f"cpu{CPU}")}
    snapshot = subprocess.run(
        ["ps", "-eo", "pid,psr,cputimes,comm", "--sort=-cputimes"],
        capture_output=True, check=True, text=True
    )
    return {"monotonic": time.monotonic(), "cpu_stat": counters, "processes": snapshot.stdout}


def sample_workers(pid: int, threads: set[int], cpus: set[str], libraries: set[str],
                   descendants: set[tuple], log: Path, capture_libraries: bool,
                   source_path: str | None, source_opened: bool, start_ns: int) -> bool:
    root = Path("/proc") / str(pid)
    try:
        raw_status = (root / "status").read_text()
        thread_count, affinity = (None, None)
        for line in raw_status.splitlines():
            key, _, value = line.partition(":\t")
            if key == "Threads":
                thread_count = int(value)
            if key == "Cpus_allowed_list":
                affinity = value
        if thread_count is not None:
            threads.add(thread_count)
        if affinity is not None:
            cpus.add(affinity)
        if capture_libraries or (source_path and not source_opened):
            for line in (root / "maps").read_text().splitlines():
                parts = line.split(None, 5)
                if len(parts) != 6:
                    continue
                if capture_libraries and parts[-1].startswith("/") and ".so" in parts[-1]:
                    libraries.add(parts[-1])
                if parts[-1] == source_path:
                    source_opened = True
        pending = [(pid, child) for child in (root / "task" / str(pid) / "children").read_text().split()]
        if not pending:
            return source_opened
        during_events = source_opened or b"Processing events " in log.read_bytes()
        for parent, child in pending:
            child_root = Path("/proc") / child
            try:
                status = dict(line.split(":\t", 1) for line in
                              (child_root / "status").read_text().splitlines() if ":\t" in line)
                argv = tuple(
                    os.fsdecode(arg) for arg in (child_root / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
                ) if source_path else ()
                elapsed_ns = time.monotonic_ns() - start_ns if source_path else 0
                descendants.add((int(child), status["Name"], status["Cpus_allowed_list"],
                                 int(status["Threads"]), during_events, parent, argv, elapsed_ns))
                pending.extend((int(child), descendant) for descendant in
                               (child_root / "task" / child / "children").read_text().split())
            except (FileNotFoundError, ProcessLookupError):
                descendants.add((int(child), "<exited>", None, 0, during_events, parent, (), 0))
    except (FileNotFoundError, ProcessLookupError):
        pass
    return source_opened


def run_leg(command: list[str], env: dict[str, str], directory: Path, label: str, *,
            event_source: Path | None = None, startup_validator=None) -> dict:
    threads, cpus, children, libraries = set(), set(), set(), set()
    before = sample_activity()
    source_path = str(event_source.resolve(strict=True)) if event_source else None
    source_opened = False
    log_path = directory / f"{label}.log"
    with log_path.open("wb") as log:
        start_ns = time.monotonic_ns()
        controller_start_ns = time.process_time_ns()
        worker = subprocess.Popen(["taskset", "-c", str(CPU), *command], cwd=directory,
                                  stdout=log, stderr=subprocess.STDOUT, env=env)
        status = None
        sample = 0
        while status is None:
            if sample % 20 == 0:
                source_opened = sample_workers(worker.pid, threads, cpus, libraries, children, log_path,
                                               sample % 100 == 0, source_path, source_opened, start_ns)
            sample += 1
            pid, code, usage = os.wait4(worker.pid, os.WNOHANG)
            if pid:
                end_ns = time.monotonic_ns()
                controller_end_ns = time.process_time_ns()
                status = os.waitstatus_to_exitcode(code)
                break
            time.sleep(0.005)
        worker.returncode = status
    after = sample_activity()
    # Initial samples can precede taskset. Only post-affinity configuration is
    # used to judge worker threading. The wrapper is itself pinned by the CLI.
    result = {
        "argv": ["taskset", "-c", str(CPU), *command],
        "wall_s": (end_ns - start_ns) / 1e9,
        "controller_cpu_s": (controller_end_ns - controller_start_ns) / 1e9,
        "cpu_user_s": usage.ru_utime,
        "cpu_sys_s": usage.ru_stime,
        "peak_rss_kb": usage.ru_maxrss,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
        "exit": status,
        "threads_seen": sorted(threads),
        "cpus_seen": sorted(cpus),
        "descendant_observations": [
            {"pid": pid, "comm": name, "cpu": affinity, "threads": count, "during_events": in_events,
             "parent_pid": parent, "argv": argv, "observed_elapsed_ns": elapsed_ns}
            for pid, name, affinity, count, in_events, parent, argv, elapsed_ns in sorted(children)
        ],
        "libraries": sorted(libraries),
        "source_mapping_observed": source_opened if event_source else None,
        "guest_before": before,
        "guest_after": after,
    }
    (directory / f"{label}.json").write_text(json.dumps(result, indent=2) + "\n")
    if status != 0:
        raise RuntimeError(f"{label} exited with {status}; see {directory}")
    unexpected_children = [
        child for child in result["descendant_observations"]
        if child["during_events"] or child["cpu"] not in {str(CPU), None} or child["threads"] > 1
        or (not (label == "framework" and child["comm"] in ROOT_DISCOVERY_COMMANDS | {"<exited>"})
            and not (startup_validator and startup_validator(child, result["descendant_observations"])))
    ]
    if threads != {1} or cpus != {str(CPU)} or unexpected_children:
        raise RuntimeError(f"{label} failed the one-CPU/one-thread check; see {directory}")
    return result


def check_pair(nominal: Path, full: Path) -> None:
    with np.load(nominal) as one, np.load(full) as many:
        np.testing.assert_array_equal(one["edges"], many["edges"])
        np.testing.assert_array_equal(one["events"], many["events"])
        for prefix in ("sr", "lr"):
            if f"{prefix}_states" not in one:
                continue
            assert one[f"{prefix}_states"].tolist() == ["NOSYS"]
            names_full = many[f"{prefix}_states"].tolist()
            assert names_full[0] == "NOSYS"
            for field in ("entries", "sumw", "sumw2"):
                np.testing.assert_array_equal(one[f"{prefix}_{field}"], many[f"{prefix}_{field}"][..., :1])


def check_quality(records: list[dict]) -> dict:
    selected = records[1:]
    measurements = [item["wall_s"] for item in selected]
    mean = statistics.mean(measurements)
    sample_sd = statistics.stdev(measurements)
    median = statistics.median(measurements)
    deviation = max(abs(value - median) for value in measurements) / median
    return {
        "warmup_s": records[0]["wall_s"],
        "retained": measurements,
        "mean_s": mean,
        "sample_sd_s": sample_sd,
        "sample_cv": sample_sd / mean,
        "max_median_deviation": deviation,
        "pass": len(measurements) >= 3 and sample_sd / mean <= MAX_COEFFICIENT_OF_VARIATION
                and deviation <= MAX_DEVIATION_FROM_MEDIAN,
    }


@APP.command()
def run(result: Path, events: int = EVENTS, pilot: bool = False,
        initial: bool = False, scope: Scope = Scope.combined) -> None:
    """Run four interleaved repetitions or an explicitly untimed correctness pilot."""
    result = result.resolve()
    result.mkdir()
    if os.sched_getaffinity(0) != {CPU}:
        raise ValueError(f"Launch the benchmark driver with taskset -c {CPU}")
    if pilot and initial:
        raise ValueError("A pilot and an initial full-file cohort are separate")
    if not pilot and events != EVENTS:
        raise ValueError(f"The production timing scope is {EVENTS} events")
    if len(list(Path("/proc/self/task").iterdir())) != 1:
        raise ValueError("Launch the controller with OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1")
    framework_env = release_environment()
    stand_env = dict(os.environ)
    limits = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                                    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
                                    "ROOT_MAX_THREADS")}
    # The stock and standalone Python processes inherit only their own runtime.
    for key in ("PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD"):
        stand_env.pop(key, None)
    stand_env.update(limits)
    release = Path(framework_env["AtlasArea"]) / "InstallArea" / framework_env["CMTCONFIG"]
    python = ROOT / ".reference-venv/bin/python"
    histogrammer = ROOT / "tools/reference/workflow.py"
    inventory = ROOT / "config/jet-states.json"
    spec = ROOT / "config/jet-histogram.json"
    provenance = {
        "input": framework_env["JETS_REFERENCE_INPUT"],
        "input_sha256": json.loads((ROOT / "config/jets-input.json").read_text())["sha256"],
        "events": events,
        "cpu": CPU,
        "thread_caps": limits,
        "analysisbase": str(release),
        "framework_python": framework_env["PYTHONHOME"],
        "histogram_python": str(python.resolve()),
        "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "clock": "monotonic wall including process startup, initialization, I/O, serialization and exit",
        "scope": "framework writes unmerged stock ntuple; separate histogram process opens that ntuple",
        "cache_policy": "warm filesystem/CVMFS",
        "exclude_first_run_per_mode": not (pilot or initial),
        "pid_sampling_s": 0.1,
        "exit_polling_s": 0.005,
        "stock_startup_utilities": sorted(ROOT_DISCOVERY_COMMANDS),
        "startup_utility_evidence": "results/reference/nominal-fork-audit-004/job.log",
        "pilot": pilot,
        "initial_full_file_cohort": initial,
        "scope_collections": scope.value,
        "guest": dict(zip(("system", "node", "release", "version", "machine"), os.uname(), strict=True)),
        "meminfo": Path("/proc/meminfo").read_text(),
        "cpuinfo": Path("/proc/cpuinfo").read_text().split("\n\n")[0],
    }
    (result / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    measurements = {mode: defaultdict(list) for mode in MODES}
    prior_hashes = {}
    for round_index in range(1 if pilot or initial else REPETITIONS):
        output = {}
        for mode in MODES[round_index % 3:] + MODES[:round_index % 3]:
            directory = result / f"{round_index + 1:02d}-{mode}"
            directory.mkdir()
            stem = "jets-stored" if mode == "stored" else "jets-a"
            suffix = "" if scope == Scope.combined else f"-{scope.value}"
            config = ROOT / "config" / f"{stem}{suffix}.yaml"
            cmd = [str(release / "bin/CPRun.py"), "-i", provenance["input"], "-t", str(config),
                   "-o", "reference.root", "-e", str(events), "--direct-driver"]
            if mode == "nominal":
                cmd.append("--no-systematics")
            frame = run_leg(cmd, framework_env, directory, "framework")
            ntuple = directory / "reference.root"
            with open_root(ntuple) as root_file:
                tree = root_file["analysis"]
                if tree.num_entries != events:
                    raise RuntimeError(f"Incomplete framework output in {directory}")
                branches = set(tree.keys())
                manifest = json.loads(inventory.read_text())
                expected_small = set(manifest["sr_p4"] if mode == "full" else ["NOSYS"])
                expected_large = set(manifest["lr_histogram"] if mode == "full" else ["NOSYS"])
                if scope == Scope.large:
                    expected_small = set()
                elif scope == Scope.small:
                    expected_large = set()
                if (branch_states(branches, "sr_pt_"), branch_states(branches, "lr_pt_")) != (
                    expected_small, expected_large
                ):
                    raise ValueError(f"Unexpected observable branch coverage in {directory}")
            with open_root(directory / "hist-reference.root") as counters:
                service = counters["systematicsList"].axis().labels()
                expected_service = expected_small | expected_large
                if mode == "full" and expected_small:
                    expected_service.update(manifest["service_weight_only"])
                if set(service) != expected_service:
                    raise ValueError(f"Unexpected independently executed states in {directory}")
                expected_order = [state for state in manifest["service_kinematic"]
                                  if state in expected_small | expected_large]
                order = [state for state in service if state not in manifest["service_weight_only"]]
                if order != expected_order:
                    raise ValueError(f"Independent kinematic state order differs in {directory}")
            if mode == "stored":
                with (directory / "framework.log").open() as logfile:
                    if any("Calibration sequence:" in line for line in logfile):
                        raise ValueError(f"Stored-object control ran a calibration in {directory}")
            output[mode] = directory / "histograms.npz"
            hist = run_leg([str(python), str(histogrammer), "histogram", str(ntuple), str(inventory),
                            str(spec), str(output[mode]), mode, "--scope", scope.value], stand_env, directory, "histogram")
            measurements[mode]["framework"].append(frame)
            measurements[mode]["histogram"].append(hist)
            measurements[mode]["total"].append({"wall_s": frame["wall_s"] + hist["wall_s"]})
            with output[mode].open("rb") as result_stream:
                digest = hashlib.file_digest(result_stream, "sha256").hexdigest()
            if mode in prior_hashes and digest != prior_hashes[mode]:
                raise RuntimeError(f"Nonreproducible result in {directory}")
            prior_hashes[mode] = digest
            print(f"{round_index + 1} {mode}: {frame['wall_s']:.3f}s + {hist['wall_s']:.3f}s",
                  flush=True)
        check_pair(output["nominal"], output["full"])

    summary = {"runs": measurements, "histogram_sha256": prior_hashes}
    if not (pilot or initial):
        summary["quality"] = {mode: {leg: check_quality(records) for leg, records in legs.items()}
                              for mode, legs in measurements.items()}
    (result / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not (pilot or initial) and not all(quality["pass"] for legs in summary["quality"].values()
                             for quality in legs.values()):
        raise typer.Exit(code=2)


if __name__ == "__main__":
    APP()
