"""Compare exact installed 006 CLI calls with narrowly borrowed NumPy arguments."""
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from zipfile import ZipFile

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packaging"))
from compare_shipment_progression import (
    BASE_ENV, CPU, EVENTS, REFERENCE, SCOPES, execute, installed, require_cpu, schedule, stock_endpoint, summarize,
)
from profile_native import digest, write_json

INSTALLATION = ROOT / "results/packaging/pristine-006"
BUNDLE = ROOT / "results/packaging/bundle-006"
FIXTURE = Path(__file__).with_name("contiguous_argument_trial_fixture.py")
ANCHOR = "\n    fargs = (arg.flatten() for arg in bargs)\n"
IMPORT = "def _evaluate(\n"
TREATMENT = '''# Private qualification switch for the exact single-worker demo.
import atexit as _trial_atexit
from pathlib import Path as _trial_path
import os as _trial_os
import sys as _trial_sys

_trial_entry = _trial_os.environ.get("ATLAS_JET_CONTIGUOUS_DEMO")
_trial_active = bool(
    _trial_entry
    and _trial_path(_trial_sys.argv[0]).resolve() == _trial_path(_trial_entry).resolve()
    and len(list(_trial_path("/proc/self/task").iterdir())) == 1
)
_trial_log = _trial_os.environ.get("ATLAS_JET_CONTIGUOUS_COUNTS") if _trial_active else None
_trial_counts = {
    "batches": 0, "batches_with_borrow": 0, "borrowed_arrays": 0, "fallback_arrays": 0,
    "borrowed_bytes": 0, "fallback_bytes": 0, "borrowed_dtype_arrays": {},
}


def _trial_borrowable(arg):
    # The selected provider consumes const data synchronously but releases
    # the GIL. Only the sealed, one-worker CLI owns this input lifetime.
    return (type(arg) is numpy.ndarray and arg.ndim == 1 and arg.flags.c_contiguous
            and arg.flags.owndata and arg.dtype.isnative and arg.dtype.kind in "fiu")


def _trial_counted_arguments(arguments):
    flags = [_trial_borrowable(arg) for arg in arguments]
    _trial_counts["batches"] += 1
    _trial_counts["batches_with_borrow"] += any(flags)
    for arg, borrowed in zip(arguments, flags, strict=True):
        kind = "borrowed" if borrowed else "fallback"
        _trial_counts[f"{kind}_arrays"] += 1
        _trial_counts[f"{kind}_bytes"] += arg.nbytes
        if borrowed:
            dtype = arg.dtype.name
            dtypes = _trial_counts["borrowed_dtype_arrays"]
            dtypes[dtype] = dtypes.get(dtype, 0) + 1
    return (arg if borrowed else arg.flatten() for arg, borrowed in zip(arguments, flags, strict=True))


if _trial_log:
    @_trial_atexit.register
    def _trial_write_counts():
        import json as _trial_json
        with open(_trial_log, "x", encoding="utf-8") as stream:
            _trial_json.dump(_trial_counts, stream, indent=2)
            stream.write("\\n")


'''
FLATTEN = '''
    if _trial_log:
        fargs = _trial_counted_arguments(bargs)
    elif _trial_active:
        fargs = (arg if _trial_borrowable(arg) else arg.flatten() for arg in bargs)
    else:
        fargs = (arg.flatten() for arg in bargs)
'''


def metadata(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def assets(package):
    return {path.relative_to(package).as_posix(): digest(path)
            for path in sorted(package.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts}


def copy_asset(source, target):
    source, target = Path(source), Path(target)
    if ".so" in source.name:
        target.symlink_to(source.resolve(strict=True))
        return str(target)
    return shutil.copyfile(source, target)


def shadows(arm, output):
    package = Path(arm["packaged"]["native"]["libcorrectionlib.so"]["path"]).parents[1]
    original = assets(package)
    with ZipFile(arm["packaged"]["path"]) as wheel:
        entries = [member for member in wheel.namelist() if member.startswith("correctionlib/")
                   and not member.endswith("/")]
        shipped = {member.removeprefix("correctionlib/"): sha256(wheel.read(member)).hexdigest()
                   for member in entries}
    if shipped != original:
        raise ValueError("The installed CorrectionLib Python package differs from wheel 006")
    prepared = {}
    for name in ("control", "borrow"):
        directory = output / f"{name}-python"
        shadow = directory / "correctionlib"
        shutil.copytree(package, shadow, ignore=shutil.ignore_patterns("__pycache__"), copy_function=copy_asset)
        highlevel = shadow / "highlevel.py"
        if name == "borrow":
            source = highlevel.read_text()
            if source.count(ANCHOR) != 1 or source.count(IMPORT) != 1:
                raise ValueError("The installed high-level argument preparation changed")
            highlevel.write_text(source.replace(IMPORT, TREATMENT + IMPORT).replace(ANCHOR, FLATTEN))
        actual = assets(shadow)
        changed = {key for key in actual if actual[key] != original.get(key)}
        if actual.keys() != original.keys() or changed != ({"highlevel.py"} if name == "borrow" else set()):
            raise ValueError(f"Unexpected checked Python package changes for {name}")
        env = {**BASE_ENV, "PYTHONPATH": str(directory), "ATLAS_JET_CONTIGUOUS_DEMO": str(arm["script"])}
        prepared[name] = {**arm, "env": env, "shadow": shadow, "shadow_members": actual}
    return prepared


def check_fixtures(arms, output):
    results = {}
    for label, name, active in (("control", "control", False), ("borrow", "borrow", True),
                                ("unguarded", "borrow", False)):
        arm = arms[name]
        env = dict(arm["env"])
        if label == "unguarded":
            env.pop("ATLAS_JET_CONTIGUOUS_DEMO")
        command = [str(arm["python"]), str(FIXTURE), str(arm["script"]), "true" if active else "false"]
        with (output / f"{label}-fixture.log").open("x") as log:
            result = subprocess.run(command, env=env, cwd=output, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Fixture failed for {label}")
        receipt = json.loads((output / f"{label}-fixture.log").read_text())
        if receipt["highlevel"] != str(arm["shadow"] / "highlevel.py"):
            raise ValueError(f"Unexpected Python import route for {label}")
        binding = next(member for name, member in arm["packaged"]["native"].items() if name.startswith("_core."))
        mapped = set(receipt["runtime_maps"])
        if receipt["binding_real_path"] != binding["path"] or not {
            member["path"] for member in arm["packaged"]["native"].values()
        } <= mapped:
            raise ValueError(f"Imported evaluator or native runtime differs for {label}")
        results[label] = receipt
    if len({row["output_sha256"] for row in results.values()}) != 1:
        raise ValueError("Focused evaluations differ between the exact copied and guarded routes")
    write_json(output / "fixture-summary.json", results)


def main(output: Path, eligibility: bool = True, repeat: bool = False):
    """Run one optional pair diagnostic or a separate bounded repeat against exact 006."""
    require_cpu()
    output = output.resolve()
    output.mkdir()
    source = json.loads((ROOT / "config/jets-input.json").read_text())
    original = Path(source["path"]).resolve(strict=True)
    observed = json.loads((REFERENCE / "full-clean-001/verification.json").read_text())
    if (digest(original) != source["sha256"] or observed["source_file_uuid"] != source["tfile_uuid"]
            or observed["events"] != EVENTS):
        raise ValueError("Original source or independent observer differs from the sealed workload")
    objects = {key: value["objects"] for key, value in observed["collections"].items()}
    authorities = {
        scope: stock_endpoint(REFERENCE / f"full-file-pilot-{('combined' if scope == 'both' else scope)}-002",
                              scope, original, source, digest) for scope in SCOPES
    }
    arm = installed(INSTALLATION, BUNDLE, output / "installed-elf")
    arms = shadows(arm, output)
    jobs = list(schedule(tuple(arms)))
    write_json(output / "provenance.json", {
        "base": arm["provenance"], "original": source, "authority": authorities,
        "controller": metadata(Path(__file__)), "fixture": metadata(FIXTURE),
        "schedule": jobs, "status": "separate bounded repeat" if repeat else "initial engineering screen",
        "candidate_policy": "Borrow only plain owned contiguous 1D arrays in the exact opted-in, single-worker CLI. "
                            "The native provider reads synchronously and has no Python callbacks. General API uses flatten.",
        "shadow_paths": {name: {"package": str(arm["shadow"]), "environment": arm["env"],
                                "members": arm["shadow_members"]} for name, arm in arms.items()},
    })
    with (ROOT / f"results/.benchmark-cpu{CPU}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_fixtures(arms, output)
        if eligibility:
            coverage = output / "eligibility"
            coverage.mkdir()
            scope = "both"
            counts_path = coverage / "arrays.json"
            measured = {**arms["borrow"], "env": {**arms["borrow"]["env"],
                        "ATLAS_JET_CONTIGUOUS_COUNTS": str(counts_path)}}
            execute("borrow", measured, scope, original, source, objects, authorities[scope]["histogram"], coverage)
            counts = json.loads(counts_path.read_text())
            total = counts["borrowed_arrays"] + counts["fallback_arrays"]
            if counts["batches"] <= 0 or total <= 0:
                raise ValueError(f"Missing eligibility data for {scope}")
            write_json(coverage / "summary.json", {"scope": scope, "arrays": counts,
                       "fallback_fraction": counts["fallback_arrays"] / total,
                       "timing_classification": "instrumented eligibility only"})
        evidence = {scope: {name: [] for name in arms} for scope in SCOPES}
        branches = {}
        for round_no, scope, name in jobs:
            work = output / f"{round_no:02d}-{scope}-{name}"
            work.mkdir()
            row = execute(name, arms[name], scope, original, source, objects, authorities[scope]["histogram"], work)
            branch = row["branch_sha256"]
            if branches.setdefault(scope, branch) != branch:
                raise ValueError(f"Physical input differs between the control and the candidate for {scope}")
            evidence[scope][name].append(row)
    results = summarize(evidence)
    for scope, result in results.items():
        control = result["arms"]["control"]["wall"]
        treatment = result["arms"]["borrow"]["wall"]
        result["same_round_wall_ratios"] = [left / right for left, right in zip(
            control["retained"], treatment["retained"], strict=True)]
        result["control_over_borrow"] = control["mean_s"] / treatment["mean_s"]
        result["qualified_for_spread"] = control["pass"] and treatment["pass"]
        result.pop("speedup_ratios")
    write_json(output / "summary.json", {"all_histograms_exact": True, "matched_physical_branches": branches,
                                        "results": results, "all_series_within_spread":
                                        all(row["qualified_for_spread"] for row in results.values())})


if __name__ == "__main__":
    typer.run(main)
