"""Record the installed standalone jet example over complete original-input passes."""

from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import platform
import re
from selectors import DefaultSelector, EVENT_READ
import signal
import subprocess
import time
from typing import Annotated
from zipfile import ZipFile

import typer

PASS = re.compile(r"^Pass (\d+)/(\d+), (\d+) events, (.*)$")
NATIVE = {"libcorrectionlib.so", "libjets.so", "libstdc++.so.6", "libgcc_s.so.1"}
DATA = {"jets.json", "jet-states.json", "jet-histogram.json", "provenance.json", "build.json"}
FORBIDDEN = (
    "libROOT", "libCore.so", "libRIO.so", "libHist.so", "libTree.so", "libCling.so",
    "libMathCore.so", "libcppyy", "libxAOD", "libJetCalibTools", "libJetUncertainties", "liblwtnn",
)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def wheel_identity(wheel, python):
    # Keep the venv path instead of resolving its executable symlink.
    packages = python.absolute().parent.parent / "lib/python3.14/site-packages"
    if not packages.is_dir():
        raise ValueError(f"Expected the supplied wheel's pristine CPython 3.14 venv: {python}")
    native, data = {}, {}
    with ZipFile(wheel) as archive:
        for member in archive.namelist():
            name = Path(member).name
            is_binding = name.startswith("_core.cpython-314") and name.endswith(".so")
            is_native = name in NATIVE or is_binding
            is_data = member.startswith("atlas_jet_demo/data/") and name in DATA
            if not (is_native or is_data):
                continue
            path = (packages / member).resolve(strict=True)
            archive_sha = sha256(archive.read(member)).hexdigest()
            if digest(path) != archive_sha:
                raise ValueError(f"Installed bytes differ from wheel member {member}")
            selected = native if is_native else data
            if name in selected:
                raise ValueError(f"Duplicate wheel member {name}")
            selected[name] = {"path": str(path), "sha256": archive_sha}
    if not NATIVE <= native.keys() or len(native) != len(NATIVE) + 1:
        raise ValueError("Missing or unexpected native wheel members")
    if set(data) != DATA:
        raise ValueError("Missing wheel calibration or analysis assets")
    return {"path": str(wheel.resolve()), "sha256": digest(wheel), "native": native, "data": data}


def proc_worker(perf_pid):
    try:
        children = Path(f"/proc/{perf_pid}/task/{perf_pid}/children").read_text().split()
    except FileNotFoundError:
        return None
    if len(children) > 1:
        raise RuntimeError(f"Expected one profiled workload, observed child PIDs {children}")
    return int(children[0]) if children else None


def snapshot(pid):
    if pid is None:
        return None
    root = Path(f"/proc/{pid}")
    try:
        status = dict(line.split(":\t", 1) for line in (root / "status").read_text().splitlines()
                      if ":\t" in line)
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        ticks = os.sysconf("SC_CLK_TCK")
        now = time.monotonic_ns()
        return {"time_ns": now, "pid": pid, "threads": int(status["Threads"]),
                "cpus": status["Cpus_allowed_list"], "user_ticks": int(fields[11]),
                "system_ticks": int(fields[12]), "ticks_per_second": ticks}
    except (FileNotFoundError, ProcessLookupError):
        return None


def run_profile(command, env, output):
    started = time.monotonic_ns()
    markers, samples = [], []
    latest_sample_ns = 0
    status = None
    pid = None
    with (output / "application.log").open("xb") as log, (output / "perf.log").open("xb") as err:
        job = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=err, env=env, start_new_session=True)
        try:
            with DefaultSelector() as selector:
                selector.register(job.stdout, EVENT_READ)
                pending = b""
                while True:
                    now = time.monotonic_ns()
                    if now - latest_sample_ns >= 500_000_000:
                        pid = pid or proc_worker(job.pid)
                        process = snapshot(pid)
                        if process is not None:
                            samples.append(process)
                        latest_sample_ns = now
                    if not selector.select(timeout=0.25):
                        continue
                    received = os.read(job.stdout.fileno(), 65536)
                    if not received:
                        break
                    log.write(received)
                    *lines, pending = (pending + received).split(b"\n")
                    for line in lines:
                        match = PASS.fullmatch(line.decode("utf-8", errors="replace"))
                        if not match:
                            continue
                        marker_ns = time.monotonic_ns()
                        pid = pid or proc_worker(job.pid)
                        markers.append({"pass": int(match[1]), "of": int(match[2]), "events": int(match[3]),
                                        "objects": match[4], "controller_monotonic_ns": marker_ns,
                                        "wall_from_launch_s": (marker_ns - started) * 1e-9})
                status = job.wait()
        except BaseException:
            if job.poll() is None:
                os.killpg(job.pid, signal.SIGTERM)
                try:
                    job.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(job.pid, signal.SIGKILL)
                    job.wait()
            raise
    return {"started_ns": started, "return_code": status, "pid": pid, "pass_ends": markers, "proc_samples": samples}


def processing_window(run, repeat, events, minimum):
    markers = run["pass_ends"]
    if run["return_code"] or len(markers) != repeat or run["pid"] is None:
        raise RuntimeError(f"perf status {run['return_code']}; observed {len(markers)}/{repeat} complete passes")
    objects = markers[0]["objects"]
    for index, marker in enumerate(markers, 1):
        if (marker["pass"], marker["of"], marker["events"], marker["objects"]) != (index, repeat, events, objects):
            raise ValueError(f"Inconsistent complete-pass marker: {marker}")
    start, end = (markers[index]["controller_monotonic_ns"] for index in (0, -1))
    samples = run["proc_samples"]
    if not samples or any(sample["threads"] != 1 or sample["cpus"] != "3" for sample in samples):
        raise RuntimeError("A sampled demo worker was missing the required one-thread CPU3 affinity")
    # Drop a full second before the final observed stdout marker to exclude
    # buffered delivery and the final NPZ write from the independent CPU floor.
    interior = [row for row in samples if start <= row["time_ns"] < end - 1_000_000_000]
    if len(interior) < 2:
        raise RuntimeError("Not enough worker CPU snapshots inside the passes 2..N processing interval")
    first, last = interior[0], interior[-1]
    hz = first["ticks_per_second"]
    # Each /proc counter is quantized to a tick. Subtract one tick per
    # counter when computing the conservative scheduling-time lower bound.
    user = max(0, last["user_ticks"] - first["user_ticks"] - 1) / hz
    system = max(0, last["system_ticks"] - first["system_ticks"] - 1) / hz
    if user + system < minimum:
        raise RuntimeError(f"Observed CPU-time floor {(user + system):.2f}s is below {minimum:g}s; use more complete passes")
    return {"clock": "CLOCK_MONOTONIC", "start_ns": start, "end_ns": end,
            "wall_s": (end - start) * 1e-9, "first_cpu_snapshot": first, "last_cpu_snapshot": last,
            "worker_user_cpu_floor_s": user, "worker_system_cpu_floor_s": system,
            "worker_cpu_floor_s": user + system, "required_floor_s": minimum,
            "method": "observed stdout pass 1/final completion; tick-adjusted CPU floor from interior snapshots"}


def exact_histogram(authority, histogram, scope):
    import numpy as np

    collections = ("sr", "lr") if scope == "both" else ({"small": ("sr",), "large": ("lr",)}[scope])
    keys = {"edges", "events", *(f"{prefix}_{leaf}" for prefix in collections
                                for leaf in ("states", "entries", "sumw", "sumw2"))}
    with np.load(authority, allow_pickle=False) as stock, np.load(histogram, allow_pickle=False) as observed:
        if not keys <= set(stock.files) or keys != set(observed.files):
            raise ValueError("Output and stock histogram key sets do not cover the requested collection scope")
        return {key: bool(np.array_equal(observed[key], stock[key], equal_nan=False)) for key in sorted(keys)}


def verify_loaded(output, wheel):
    audit = json.loads((output / "library-maps.json").read_text())
    seen = [Path(path) for path in audit["libraries"]]
    if any(path.is_relative_to("/cvmfs") or path.name.startswith(FORBIDDEN) for path in seen):
        raise ValueError("Profile mapped a ROOT, AnalysisBase or CVMFS library")
    for name, identity in wheel["native"].items():
        expected = Path(identity["path"])
        if name.startswith("_core.cpython"):
            matches = {path for path in seen if path.parent.name == "correctionlib"
                       and path.name.startswith("_core.cpython")}
        else:
            matches = {path for path in seen if path.name.startswith(name.split(".so")[0])}
        if matches != {expected} or digest(expected) != identity["sha256"]:
            raise ValueError(f"Profile mapped a different {name}: {matches}")
    return audit


def main(
    python: Path, demo: Path, original: Path, output: Path, authority: Path,
    wheel: Annotated[Path, typer.Option(help="Exact wheel installed in the profiled interpreter")],
    repeat: Annotated[int, typer.Option(min=2)] = 6,
    scope: str = "both",
    minimum: Annotated[float, typer.Option(min=60.0)] = 60.0,
    perf: Path = Path("/usr/bin/perf"),
    record_only: bool = False,
    use_demo_defaults: bool = False,
):
    """Record native CPU stacks while profiling a complete, exact standalone workflow."""
    if scope not in ("small", "large", "both"):
        raise ValueError("Scope must be small, large, or both")
    if os.sched_getaffinity(0) != {3}:
        raise ValueError("Run the profile controller on the reserved CPU 3")
    output = output.resolve()
    output.mkdir(exist_ok=False)
    try:
        import numpy as np

        packaged = wheel_identity(wheel, python)
        defaults = None
        if use_demo_defaults:
            manifest = json.loads(wheel.with_name("bundle.json").read_text())
            if (digest(demo) != manifest["visible_script"]["sha256"]
                    or packaged["sha256"] != manifest["wheel"]["sha256"]):
                raise ValueError("Visible script or installed wheel differs from the exact shipment")
            defaults = manifest["demo_defaults"]
            if defaults["mode"] != "full" or defaults["scope"] != "both":
                raise ValueError("Recorded delivery defaults differ from the complete calibrated workflow")
        with np.load(authority, allow_pickle=False) as stock:
            events = int(stock["events"])
        machine = {
            "kernel": platform.uname()._asdict(),
            "perf": subprocess.run([str(perf), "version"], check=True, capture_output=True, text=True).stdout.strip(),
            "perf_event_paranoid": Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip(),
            "event_sources": sorted(p.name for p in Path("/sys/bus/event_source/devices").iterdir()),
            "event": "cpu-clock:u", "period_ns": 10_000_000, "clock": "CLOCK_MONOTONIC",
        }
        write_json(output / "machine.json", machine)
        with (output / "provider-elf-sections.txt").open("x", encoding="utf-8") as sections:
            subprocess.run(["readelf", "-SWn", packaged["native"]["libjets.so"]["path"]],
                           stdout=sections, stderr=subprocess.STDOUT, check=True)
        env = {
            "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1", "ATLAS_JET_LIBRARY_AUDIT": str(output / "library-maps.json"),
        }
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                     "BLIS_NUM_THREADS", "ROOT_MAX_THREADS", "RAYON_NUM_THREADS"):
            env[name] = "1"
        data = output / "record.data"
        histogram = output / "histogram.npz"
        command = [
            str(perf), "record", "--event", machine["event"], "--count", str(machine["period_ns"]),
            "--clockid", "CLOCK_MONOTONIC", "--timestamp", "--period", "--sample-cpu",
            "--no-inherit", "--no-buildid-cache", "--call-graph",
            "dwarf,16384", "-o", str(data), "--", str(python.absolute()), str(demo.resolve()), str(original.absolute()),
            str(histogram), "-n", str(repeat),
        ]
        if use_demo_defaults:
            if scope != "both":
                command.extend(["--scope", scope])
        else:
            command.extend(["--mode", "full", "--scope", scope, "--matched-read",
            "--provider", packaged["native"]["libjets.so"]["path"], "--payload", packaged["data"]["jets.json"]["path"],
            "--states", packaged["data"]["jet-states.json"]["path"],
            "--histogram-spec", packaged["data"]["jet-histogram.json"]["path"],
            "--provenance", packaged["data"]["provenance.json"]["path"],
            ])
        write_json(output / "command.json", command)
        run = run_profile(command, env, output)
        write_json(output / "process.json", run)
        interval = processing_window(run, repeat, events, minimum)
        audit = verify_loaded(output, packaged)
        checks = exact_histogram(authority, histogram, scope)
        if not all(checks.values()):
            raise ValueError(f"Histogram differs from stock: {[key for key, equal in checks.items() if not equal]}")
        if not data.is_file() or data.stat().st_size == 0:
            raise RuntimeError("No perf recording was saved")
        unique_objects = {prefix: int(count) for prefix, count in
                          re.findall(r"(sr|lr) (\d+) original jets", run["pass_ends"][0]["objects"])}
        expected_prefixes = {"both": {"sr", "lr"}, "small": {"sr"}, "large": {"lr"}}[scope]
        if unique_objects.keys() != expected_prefixes:
            raise ValueError(f"Unexpected complete-pass object counts: {unique_objects}")
        write_json(output / "recording.json", {
            "passed": True, "scope": scope, "repeat": repeat, "events_per_pass": events, "pid": run["pid"],
            "pass_ends": run["pass_ends"], "measurement": interval, "machine": machine,
            "shipped_defaults": defaults,
            "calibration_cpu": 3, "perf_collector_cpu": 3, "calibration_threads": 1,
            "unique_objects": unique_objects, "event_visits": events * repeat,
            "object_visits": {prefix: count * repeat for prefix, count in unique_objects.items()},
            "original": {"path": str(original.absolute()), "bytes": original.stat().st_size},
            "demo": {"path": str(demo.resolve()), "sha256": digest(demo.resolve())},
            "stock_histogram": {"path": str(authority.resolve()), "sha256": digest(authority)},
            "histogram": {"path": str(histogram), "sha256": digest(histogram), "exact_stock_fields": checks},
            "wheel": packaged, "audit": audit,
            "profile": {"path": str(data), "bytes": data.stat().st_size, "sha256": digest(data)},
        })
        typer.echo(f"Recorded {repeat} complete {events}-event passes, {interval['worker_cpu_floor_s']:.2f}s CPU floor.")
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        write_json(output / "failure.json", {"passed": False, "error": str(error), "time": time.time()})
        typer.echo(f"{error}; see {output}", err=True)
        raise typer.Exit(1) from error
    if not record_only:
        from native_stack_profile import summarize

        summarize(output, perf)


if __name__ == "__main__":
    typer.run(main)
