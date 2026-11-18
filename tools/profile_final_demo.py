"""Diagnostic timings for the exact shipped, installed jet demo."""

import time

WORKER_ENTRY_NS = time.monotonic_ns()
WORKER_ENTRY_CPU_NS = time.process_time_ns()

from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from importlib.machinery import PathFinder
import os
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
SCOPES = ("both", "small", "large")
COLLECTIONS = {"both": ("sr", "lr"), "small": ("sr",), "large": ("lr",)}
EXECUTION_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
                     "RAYON_NUM_THREADS", "ROOT_MAX_THREADS")


def observation():
    return {"calls": 0, "rows": 0, "wall_ns": 0, "cpu_ns": 0, "self_wall_ns": 0, "self_cpu_ns": 0}


class Measurements:
    def __init__(self):
        self.records = defaultdict(observation)
        self.batch_sizes = defaultdict(lambda: defaultdict(observation))
        self.active = []

    @contextmanager
    def observe(self, name, rows=None):
        frame = [0, 0]
        self.active.append(frame)
        wall, cpu = time.monotonic_ns(), time.process_time_ns()
        try:
            yield
        finally:
            wall, cpu = time.monotonic_ns() - wall, time.process_time_ns() - cpu
            self.active.pop()
            if self.active:
                self.active[-1][0] += wall
                self.active[-1][1] += cpu
            record = self.records[name]
            records = (record,) if rows is None else (record, self.batch_sizes[name][rows])
            for record in records:
                record["calls"] += 1
                record["rows"] += rows or 0
                record["wall_ns"] += wall
                record["cpu_ns"] += cpu
                record["self_wall_ns"] += wall - frame[0]
                record["self_cpu_ns"] += cpu - frame[1]


def vector_rows(args):
    for arg in args:
        if getattr(arg, "ndim", 0):
            return int(arg.shape[0])
    return 1


def wrapped(original, meter, name, rows=None):
    @wraps(original)
    def measure(*args, **kwargs):
        with meter.observe(name, rows(args, kwargs) if rows else None):
            return original(*args, **kwargs)
    return measure


class Tool:
    def __init__(self, original, meter, name):
        self.original = original
        self.evaluate = wrapped(original.evaluate, meter, f"tool.{name}", lambda args, _: vector_rows(args))

    def __getattr__(self, name):
        return getattr(self.original, name)


class NodeSet:
    def __init__(self, original, meter):
        self.original = original
        self.meter = meter
        self.tools = {}

    def __getitem__(self, name):
        if name not in self.tools:
            self.tools[name] = Tool(self.original[name], self.meter, name)
        return self.tools[name]

    def __getattr__(self, name):
        return getattr(self.original, name)


class Transport:
    def __init__(self, original, meter):
        self.original = original
        self.meter = meter

    def __getattr__(self, name):
        original = getattr(self.original, name)
        if name not in {"truth", "jmr_truth", "vertices", "weights"}:
            return original

        @contextmanager
        def borrow(*args, **kwargs):
            with self.meter.observe(f"transport.{name}.bind", vector_rows(args[1:])):
                lease = original(*args, **kwargs)
                value = lease.__enter__()
            try:
                yield value
            except BaseException:
                with self.meter.observe(f"transport.{name}.unbind"):
                    suppressed = lease.__exit__(*sys.exc_info())
                if not suppressed:
                    raise
            else:
                with self.meter.observe(f"transport.{name}.unbind"):
                    lease.__exit__(None, None, None)
        return borrow


def install_visible_hooks(main, meter):
    live = main.__globals__
    original_load = live["load_corrections"]

    @wraps(original_load)
    def load(*args, **kwargs):
        with meter.observe("setup.load_corrections"):
            nodes, buffers = original_load(*args, **kwargs)
        return NodeSet(nodes, meter), Transport(buffers, meter) if buffers is not None else None

    live["load_corrections"] = load

    # evaluate_chunk yields results consumed by the histogrammer. Time only
    # the periods when the original generator is executing.
    original_frames = live["evaluate_chunk"]

    @wraps(original_frames)
    def frames(*args, **kwargs):
        source = original_frames(*args, **kwargs)
        try:
            while True:
                with meter.observe("workflow.compute"):
                    try:
                        value = next(source)
                    except StopIteration:
                        return
                yield value
        finally:
            with meter.observe("workflow.close_chunk"):
                source.close()

    live["evaluate_chunk"] = frames
    for name, phase in (
        ("materialize", "setup.payload_materialization"),
        ("native_runtime", "setup.check_loaded_provider"),
        ("branch_plan", "io.branch_plan"),
        ("resolve_branches", "io.resolve_branches"),
        ("event_inputs", "workflow.event_inputs"),
        ("truth_context", "workflow.truth_inputs"),
        ("uncertainty_inputs", "workflow.uncertainty_inputs"),
        ("small_jvt_inputs", "workflow.nnjvt_inputs"),
        ("small_jvt_selection", "workflow.nnjvt_update_and_selection"),
        ("small_sf", "workflow.nnjvt_weights"),
        ("create_histogram", "histogram.initialize"),
    ):
        if name in live:
            live[name] = wrapped(live[name], meter, phase)

    def requested_entries(_, kwargs):
        start, stop = kwargs.get("entry_start"), kwargs.get("entry_stop")
        return None if start is None or stop is None else int(stop - start)

    uproot = live["uproot"]
    uproot.open = wrapped(uproot.open, meter, "io.open_phySLITE")
    directory = uproot.reading.ReadOnlyDirectory
    directory.__exit__ = wrapped(directory.__exit__, meter, "io.close_phySLITE")
    tree = uproot.behaviors.TBranch.HasBranches
    tree.arrays = wrapped(tree.arrays, meter, "io.read_decode_arrays", requested_entries)
    hist = live["hist"].Hist
    hist.fill = wrapped(hist.fill, meter, "histogram.fill",
                        lambda args, kwargs: vector_rows((*args[1:], *kwargs.values())))
    for name in ("values", "variances"):
        setattr(hist, name, wrapped(getattr(hist, name), meter, f"histogram.{name}"))
    live["np"].savez = wrapped(live["np"].savez, meter, "output.write_npz")


class HookLoader:
    def __init__(self, loader, after):
        self.loader = loader
        self.after = after

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        self.after(module)

    def __getattr__(self, name):
        return getattr(self.loader, name)


class HookImports:
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def find_spec(self, fullname, path=None, target=None):
        after = self.callbacks.get(fullname)
        if after is None:
            return None
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise ImportError(f"Required installed diagnostic module missing: {fullname}")
        spec.loader = HookLoader(spec.loader, after)
        return spec


def worker(visible, phase_output, arguments):
    meter = Measurements()
    markers = {}
    hooks = {}

    def runtime(module):
        hooks[module.__name__] = module.__file__
        for name, phase in (("prepare_runtime", "setup.compiler_runtime"), ("audit_runtime", "output.library_audit")):
            setattr(module, name, wrapped(getattr(module, name), meter, phase))

    def typer_loaded(module):
        hooks[module.__name__] = module.__file__
        original = module.run

        @wraps(original)
        def visible_run(main, *args, **kwargs):
            code = getattr(main, "__code__", None)
            if code is None or Path(code.co_filename).resolve() != visible:
                return original(main, *args, **kwargs)
            if "cli_started_ns" in markers:
                raise RuntimeError("The visible script entered its CLI twice")
            markers["cli_started_ns"] = time.monotonic_ns()
            with meter.observe("setup.instrumentation"):
                install_visible_hooks(main, meter)
            try:
                with meter.observe("visible.cli"):
                    return original(main, *args, **kwargs)
            finally:
                markers["cli_finished_ns"] = time.monotonic_ns()

        module.run = visible_run

    finder = HookImports({"atlas_jet_demo.runtime": runtime, "typer": typer_loaded})
    sys.meta_path.insert(0, finder)
    # Match direct script invocation without ever importing the checkout demo.
    sys.argv = [str(visible), *arguments]
    sys.path[0] = str(visible.parent)
    markers["visible_started_ns"] = time.monotonic_ns()
    try:
        with meter.observe("visible.total"):
            runpy.run_path(str(visible), run_name="__main__")
    finally:
        markers["visible_finished_ns"] = time.monotonic_ns()
        markers["worker_entry_ns"] = WORKER_ENTRY_NS
        markers["worker_cpu_ns"] = time.process_time_ns() - WORKER_ENTRY_CPU_NS
        sys.meta_path.remove(finder)
        import json

        with phase_output.open("x") as stream:
            json.dump({
                "argv": sys.argv, "markers": markers, "hooked_modules": hooks,
                "observations": dict(meter.records), "by_batch_rows": {
                    name: dict(rows) for name, rows in meter.batch_sizes.items()
                },
            }, stream, indent=2, allow_nan=False)
            stream.write("\n")


def describe(path, digest):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def checked_installation(python, demo, wheel, bundle_path, digest, wheel_identity):
    from hashlib import file_digest
    import json
    from zipfile import ZipFile

    bundle = json.loads(bundle_path.read_text())
    for key, path in (("wheel", wheel), ("visible_script", demo)):
        recorded = bundle[key]
        if recorded["sha256"] != digest(path) or recorded["bytes"] != path.stat().st_size:
            raise ValueError(f"Packaged {key} differs from its sealed bundle: {path}")
    packaged = wheel_identity(wheel, python)
    package = Path(packaged["data"]["build.json"]["path"]).parents[1]
    build = json.loads((package / "data/build.json").read_text())
    for key in ("payload", "provider", "git_commit", "columnar_source", "scope_index", "scope_projector",
                "demo_defaults", "visible_source"):
        if bundle.get(key) != build.get(key):
            raise ValueError(f"Installed and shipped build receipts differ for {key}")
    if any(build["visible_source"][field] != bundle["visible_script"][field] for field in ("bytes", "sha256")):
        raise ValueError("Installed wheel and shipped visible source identities differ")
    defaults = bundle.get("demo_defaults")
    if defaults is None:
        raise ValueError("Use the final wheel with a sealed demo_defaults manifest")
    if (defaults["mode"], defaults["scope"]) != ("full", "both"):
        raise ValueError("The final visible defaults need to be the frozen full combined workload")
    if not set(defaults["compact_payload_default_scopes"]) <= {"small", "large"}:
        raise ValueError("Unexpected default compact payload scope")
    source_paths = ["__init__.py", "inputs.py", "transport.py", "runtime.py"]
    if "scope_index" in bundle:
        source_paths.extend(("payload_ranges.py", "data/payload-index.json"))
    with ZipFile(wheel) as archive:
        for name in source_paths:
            path = package / name
            with archive.open(f"atlas_jet_demo/{name}") as stream:
                expected = file_digest(stream, "sha256").hexdigest()
            record = describe(path, digest)
            if record["sha256"] != expected:
                raise ValueError(f"Installed visible module differs from wheel: {path}")
            packaged.setdefault("workflow", {})[name] = record
    if "scope_index" in bundle:
        for key, name in (("scope_index", "data/payload-index.json"), ("scope_projector", "payload_ranges.py")):
            if any(bundle[key][field] != packaged["workflow"][name][field] for field in ("bytes", "sha256")):
                raise ValueError(f"Installed {key} differs from the final bundle")
    return packaged, bundle


def stock_endpoint(receipt, scope, original, source, digest):
    import json

    work = receipt / "01-full"
    provenance = receipt / "provenance.json"
    record = json.loads(provenance.read_text())
    stock_scope = "combined" if scope == "both" else scope
    if (record.get("path", "A") != "A" or record["scope_collections"] != stock_scope
            or record["events"] != source["entry_interval"][1]
            or Path(record["input"]).resolve() != original.resolve()
            or record["input_sha256"] != source["sha256"]):
        raise ValueError(f"Stock receipt is outside the exact full source and scope: {receipt}")
    legs = {}
    for leg in ("framework", "histogram"):
        path = work / f"{leg}.json"
        observed = json.loads(path.read_text())
        if observed["exit"] or observed["threads_seen"] != [1] or observed["cpus_seen"] != ["3"]:
            raise ValueError(f"Stock full-file endpoint has no successful {leg} receipt: {receipt}")
        if leg == "framework" and any(Path(p).name == "libjets.so" for p in observed["libraries"]):
            raise ValueError(f"Stock receipt mapped the converted provider: {receipt}")
        legs[leg] = describe(path, digest)
    return {"histogram": describe(work / "histograms.npz", digest),
            "provenance": describe(provenance, digest), "legs": legs}


def histogram_comparison(stock, candidate, scope, events):
    import numpy as np

    fields = {"edges", "events", *(f"{prefix}_{leaf}" for prefix in COLLECTIONS[scope]
                                  for leaf in ("states", "entries", "sumw", "sumw2"))}
    with np.load(stock, allow_pickle=False) as a, np.load(candidate, allow_pickle=False) as c:
        if set(a.files) != fields or set(c.files) != fields or int(a["events"]) != events:
            raise ValueError("The stored stock and visible histogram inventories or event interval differ")
        return {name: a[name].dtype == c[name].dtype and a[name].shape == c[name].shape
                and a[name].tobytes() == c[name].tobytes() for name in sorted(fields)}


def validate_run(directory, scope, events, source_uuid, bundle, packaged):
    import json
    import re

    from profile_native import PASS, verify_loaded

    log = (directory / "visible.log").read_text()
    passes = [record.groups() for line in log.splitlines() if (record := PASS.fullmatch(line))]
    if len(passes) != 1 or passes[0][:3] != ("1", "1", str(events)):
        raise ValueError(f"Expected one complete {events}-event pass for {scope}")
    objects = {prefix: int(count) for prefix, count in re.findall(r"(sr|lr) (\d+) original jets", passes[0][3])}
    if set(objects) != set(COLLECTIONS[scope]) or f"({source_uuid})" not in log:
        raise ValueError(f"Unexpected objects or selected source for {scope}")
    summary = f"{events} unique events, {events} event visits across 1 passes"
    if log.count(summary) != 1:
        raise ValueError(f"Unexpected event visits for {scope}")
    branch, = re.findall(r"(?m)^Input branch SHA-256: ([0-9a-f]{64})$", log)
    digest = re.findall(r"(?m)^Payload materialized SHA-256: ([0-9a-f]{64})$", log)
    sizes = re.findall(r"(?m)^Payload materialized bytes: (\d+)$", log)
    defaults = bundle["demo_defaults"]
    compact = scope in defaults["compact_payload_default_scopes"]
    actual_payload = None
    if compact:
        layout = json.loads(Path(packaged["workflow"]["data/payload-index.json"]["path"]).read_text())
        expected = layout["scopes"][scope]
        if digest != [expected["sha256"]] or sizes != [str(expected["bytes"])]:
            raise ValueError(f"Default compact payload bytes differ for {scope}")
        actual_payload = {"sha256": digest[0], "bytes": int(sizes[0])}
    elif digest or sizes:
        raise ValueError(f"Unexpected compact payload materialization for {scope}")
    reuse = bool(defaults["reuse_nominal_nnjvt_update"] and
                 scope in defaults["reuse_nominal_nnjvt_update_scopes"] and
                 defaults["mode"] in defaults["reuse_nominal_nnjvt_update_modes"])
    sign_pairs = scope in defaults.get("reuse_uncertainty_sign_pair_scopes", [])
    sign_marker = "Trial enabled: reuse unsigned uncertainty aggregate within adjacent state pairs"
    if log.count(sign_marker) != int(sign_pairs):
        raise ValueError(f"Signed-state usage differs from the sealed default for {scope}")
    reuse_marker = "Trial enabled: reuse nominal NNJvt update for bitwise-identical input rows"
    if log.count(reuse_marker) != int(reuse):
        raise ValueError(f"The packaged NNJvt reuse default did not execute as sealed for {scope}")
    return {
        "objects": objects, "branch_sha256": branch, "compact_payload": actual_payload,
        "nnjvt_reuse": reuse, "unsigned_pair_reuse": sign_pairs, "runtime_maps": verify_loaded(directory, packaged),
    }


def timing_summary(process, worker_report, events, compact):
    phases = worker_report["observations"]
    marks = worker_report["markers"]
    for key in ("visible.total", "visible.cli", "setup.compiler_runtime", "setup.load_corrections",
                "io.open_phySLITE", "io.read_decode_arrays", "histogram.fill", "output.write_npz"):
        if key not in phases:
            raise ValueError(f"Visible diagnostic did not observe {key}")
    if phases["io.read_decode_arrays"]["rows"] != events:
        raise ValueError("Observed ROOT array reads did not cover one exact event interval")
    if int("setup.payload_materialization" in phases) != int(compact):
        raise ValueError("Observed setup does not agree with the sealed compact default")
    tools = {name.removeprefix("tool."): {**values, "by_batch_rows": worker_report["by_batch_rows"][name]}
             for name, values in phases.items() if name.startswith("tool.")}
    if not tools:
        raise ValueError("No numerical correction invocation was measured")

    wall = round(process["wall_s"] * 1_000_000_000)
    live = marks["visible_finished_ns"] - marks["worker_entry_ns"]
    if live > wall:
        raise ValueError("Worker clock does not fit inside the controller's process interval")
    groups = defaultdict(int)
    # Each nested phase contributes only its exclusive time. Root visible.total
    # includes module import and the runpy finally block. Break those out with
    # worker markers and report all other diagnostic overhead as remainder.
    for name, record in phases.items():
        if name not in {"visible.total", "visible.cli", "setup.compiler_runtime"}:
            groups[name.split(".", 1)[0]] += record["self_wall_ns"]
    groups["workflow"] += phases["visible.cli"]["self_wall_ns"]
    startup = marks["cli_started_ns"] - marks["worker_entry_ns"]
    teardown = marks["visible_finished_ns"] - marks["cli_finished_ns"]
    remainder = wall - startup - teardown - sum(groups.values())
    if remainder < 0:
        raise ValueError("The measured phase partition double counted wall time")
    return {
        "external_wall_ns": wall,
        "worker_cpu_ns": marks["worker_cpu_ns"],
        "diagnostic_report_and_uninstrumented_process_remainder_ns": wall - live,
        "startup_through_visible_cli_entry_ns": startup,
        "teardown_after_visible_cli_ns": teardown,
        "startup_detail": {
            "worker_bootstrap_before_visible_ns": marks["visible_started_ns"] - marks["worker_entry_ns"],
            "visible_import_to_cli_ns": marks["cli_started_ns"] - marks["visible_started_ns"],
            "compiler_runtime_inside_visible_import_ns": phases["setup.compiler_runtime"]["wall_ns"],
        },
        "exclusive_groups_inside_visible_cli_ns": dict(groups),
        "remaining_phase_overhead_ns": remainder,
        "tools": tools,
    }


def main(python: Path, demo: Path, wheel: Path, original: Path, output: Path,
         stock_both: Path | None = None, stock_small: Path | None = None, stock_large: Path | None = None,
         bundle: Path | None = None):
    """Diagnose the installed final visible script once per requested stock-matched collection scope."""
    import fcntl
    import json

    from profile_native import digest, wheel_identity, write_json

    sys.path.insert(0, str(ROOT / "tools/reference"))
    from benchmark import CPU, run_leg

    if os.sched_getaffinity(0) != {CPU} or len(list(Path("/proc/self/task").iterdir())) != 1:
        raise ValueError(f"Launch the diagnostic controller on reserved CPU {CPU}, with one thread")
    requested = dict(zip(SCOPES, (stock_both, stock_small, stock_large), strict=True))
    requested = {scope: path.resolve(strict=True) for scope, path in requested.items() if path is not None}
    if not requested:
        raise ValueError("Supply a true full-file stock run directory for each requested scope")
    # Keep the venv executable name. Resolving its symlink would lose pyvenv.cfg.
    python = python.absolute()
    demo, wheel, original = (path.resolve(strict=True) for path in (demo, wheel, original))
    bundle_path = (bundle or wheel.with_name("bundle.json")).resolve(strict=True)
    output = output.resolve()
    with (ROOT / f"results/.benchmark-cpu{CPU}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = json.loads((ROOT / "config/jets-input.json").read_text())
        first, events = source["entry_interval"]
        if first != 0 or str(original) != str(Path(source["path"]).resolve()) or digest(original) != source["sha256"]:
            raise ValueError("The original PHYSLITE input differs from the frozen full-file source")
        packaged, metadata = checked_installation(python, demo, wheel, bundle_path, digest, wheel_identity)
        authorities = {scope: stock_endpoint(path, scope, original, source, digest) for scope, path in requested.items()}
        output.mkdir()
        write_json(output / "provenance.json", {
            "measurement": "instrumented diagnostic, separate from retained timings",
            "cpu": CPU, "threads": 1, "original": source, "events": events, "repeat": 1,
            "demo": describe(demo, digest), "wheel": packaged,
            "bundle": describe(bundle_path, digest), "controller": describe(Path(__file__).resolve(), digest),
            "selected_defaults": metadata["demo_defaults"], "stock": authorities,
            "launch": "unmodified shipped demo bytes executed as __main__ in supplied installed interpreter",
            "instrumentation": "Python boundary timings; tool times include their real evaluator dispatch",
        })
        summaries = {}
        env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1",
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
               **dict.fromkeys(EXECUTION_THREADS, "1")}
        for scope, stock in authorities.items():
            directory = output / scope
            directory.mkdir()
            phases = directory / "phase-worker.json"
            histogram = directory / "histograms.npz"
            args = [str(original), str(histogram), "--events", str(events)]
            if scope != "both":
                args.extend(("--scope", scope))
            command = [str(python), str(Path(__file__).resolve()), "--phase-worker", str(demo), str(phases), *args]
            write_json(directory / "command.json", {"process": command, "visible_argv": [str(demo), *args]})
            process = run_leg(command, {**env, "ATLAS_JET_LIBRARY_AUDIT": str(directory / "library-maps.json")},
                              directory, "visible", event_source=original)
            if not process["source_mapping_observed"]:
                raise ValueError(f"Physical input mapping was not observed for {scope}")
            reported = validate_run(directory, scope, events, source["tfile_uuid"], metadata, packaged)
            stock_histogram = Path(stock["histogram"]["path"])
            if digest(stock_histogram) != stock["histogram"]["sha256"]:
                raise ValueError(f"The {scope} stock histogram changed during the diagnostic")
            checked = histogram_comparison(stock_histogram, histogram, scope, events)
            write_json(directory / "histogram-comparison.json", {"exact_fields": checked, "actual": describe(histogram, digest)})
            if not all(checked.values()):
                raise ValueError(f"The final visible {scope} histograms differ from true stock")
            measured = json.loads(phases.read_text())
            if measured["argv"] != [str(demo), *args]:
                raise ValueError(f"Actual effective visible argv differs for {scope}")
            expected_runtime = Path(packaged["workflow"]["runtime.py"]["path"]).resolve()
            if Path(measured["hooked_modules"]["atlas_jet_demo.runtime"]).resolve() != expected_runtime:
                raise ValueError(f"The profiler hooked a different installed runtime for {scope}")
            summaries[scope] = {
                **timing_summary(process, measured, events, reported["compact_payload"] is not None),
                **reported, "exact_stock_histogram": True,
            }
            write_json(directory / "summary.json", summaries[scope])
        write_json(output / "summary.json", summaries)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--phase-worker"]:
        worker(Path(sys.argv[2]).resolve(strict=True), Path(sys.argv[3]), sys.argv[4:])
    else:
        import typer

        typer.run(main)
