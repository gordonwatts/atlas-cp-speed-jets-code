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

from workflow import open_root

APP = typer.Typer(add_completion=False)
ROOT = Path(__file__).resolve().parents[2]
MODES = ("stored", "nominal", "full")
EVENTS = 20000
CPU = 3
REPETITIONS = 4
MAX_COEFFICIENT_OF_VARIATION = 0.05
MAX_DEVIATION_FROM_MEDIAN = 0.10


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


def sample_worker(pid: int, threads: set[int], cpus: set[str], libraries: set[str],
                  children: set[int], capture_libraries: bool):
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
        if capture_libraries:
            for line in (root / "maps").read_text().splitlines():
                parts = line.split(None, 5)
                if len(parts) == 6 and parts[-1].startswith("/") and ".so" in parts[-1]:
                    libraries.add(parts[-1])
        children.update(int(value) for value in (root / "task" / str(pid) / "children").read_text().split())
    except (FileNotFoundError, ProcessLookupError):
        return


def run_leg(command: list[str], env: dict[str, str], directory: Path, label: str) -> dict:
    threads, cpus, children, libraries = set(), set(), set(), set()
    before = sample_activity()
    with (directory / f"{label}.log").open("wb") as log:
        start_ns = time.monotonic_ns()
        worker = subprocess.Popen(["taskset", "-c", str(CPU), *command], cwd=directory,
                                  stdout=log, stderr=subprocess.STDOUT, env=env)
        status = None
        sample = 0
        while status is None:
            sample_worker(worker.pid, threads, cpus, libraries, children, sample % 20 == 0)
            sample += 1
            pid, code, usage = os.wait4(worker.pid, os.WNOHANG)
            if pid:
                end_ns = time.monotonic_ns()
                status = os.waitstatus_to_exitcode(code)
                break
            time.sleep(0.025)
        worker.returncode = status
    after = sample_activity()
    # Initial samples can precede taskset. Only post-affinity configuration is
    # used to judge worker threading. The wrapper is itself pinned by the CLI.
    result = {
        "argv": command,
        "wall_s": (end_ns - start_ns) / 1e9,
        "cpu_user_s": usage.ru_utime,
        "cpu_sys_s": usage.ru_stime,
        "peak_rss_kb": usage.ru_maxrss,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
        "exit": status,
        "threads_seen": sorted(threads),
        "cpus_seen": sorted(cpus),
        "child_pids_seen": sorted(children),
        "libraries": sorted(libraries),
        "guest_before": before,
        "guest_after": after,
    }
    (directory / f"{label}.json").write_text(json.dumps(result, indent=2) + "\n")
    if status != 0:
        raise RuntimeError(f"{label} exited with {status}; see {directory}")
    if threads != {1} or cpus != {str(CPU)} or children:
        raise RuntimeError(f"{label} failed the one-CPU/one-thread check; see {directory}")
    return result


def check_pair(nominal: Path, full: Path) -> None:
    with np.load(nominal) as one, np.load(full) as many:
        np.testing.assert_array_equal(one["edges"], many["edges"])
        np.testing.assert_array_equal(one["events"], many["events"])
        for prefix in ("sr", "lr"):
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
def run(result: Path, events: int = EVENTS, pilot: bool = False) -> None:
    """Run four interleaved repetitions or an explicitly untimed correctness pilot."""
    result = result.resolve()
    result.mkdir()
    if os.sched_getaffinity(0) != {CPU}:
        raise ValueError(f"Launch the benchmark driver with taskset -c {CPU}")
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
        "events": events,
        "cpu": CPU,
        "thread_caps": limits,
        "analysisbase": str(release),
        "framework_python": framework_env["PYTHONHOME"],
        "histogram_python": str(python.resolve()),
        "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "clock": "monotonic wall including process startup, initialization, I/O, serialization and exit",
        "scope": "framework writes unmerged stock ntuple; separate histogram process opens that ntuple",
        "cache_policy": "warm filesystem/CVMFS; exclude the first run in each mode",
        "pilot": pilot,
        "guest": os.uname()._asdict(),
        "meminfo": Path("/proc/meminfo").read_text(),
        "cpuinfo": Path("/proc/cpuinfo").read_text().split("\n\n")[0],
    }
    (result / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    measurements = {mode: defaultdict(list) for mode in MODES}
    prior_hashes = {}
    for round_index in range(1 if pilot else REPETITIONS):
        output = {}
        for mode in MODES[round_index % 3:] + MODES[:round_index % 3]:
            directory = result / f"{round_index + 1:02d}-{mode}"
            directory.mkdir()
            config = ROOT / "config" / ("jets-stored.yaml" if mode == "stored" else "jets-a.yaml")
            cmd = [str(release / "bin/CPRun.py"), "-i", provenance["input"], "-t", str(config),
                   "-o", "reference.root", "-e", str(events), "--direct-driver"]
            if mode == "nominal":
                cmd.append("--no-systematics")
            frame = run_leg(cmd, framework_env, directory, "framework")
            ntuple = directory / "reference.root"
            with open_root(ntuple) as root_file:
                if root_file["analysis"].num_entries != events:
                    raise RuntimeError(f"Incomplete framework output in {directory}")
            output[mode] = directory / "histograms.npz"
            hist = run_leg([str(python), str(histogrammer), "histogram", str(ntuple), str(inventory),
                            str(spec), str(output[mode]), mode], stand_env, directory, "histogram")
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
    if not pilot:
        summary["quality"] = {mode: {leg: check_quality(records) for leg, records in legs.items()}
                              for mode, legs in measurements.items()}
    (result / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not pilot and not all(quality["pass"] for legs in summary["quality"].values()
                             for quality in legs.values()):
        raise typer.Exit(code=2)


if __name__ == "__main__":
    APP()
