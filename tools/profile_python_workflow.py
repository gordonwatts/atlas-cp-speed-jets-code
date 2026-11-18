"""One cProfile diagnostic of the installed, stock-matched combined jet demo."""

import fcntl
import json
import os
from pathlib import Path
import pstats
import sys

import typer

from profile_final_demo import (
    EXECUTION_THREADS, ROOT, checked_installation, histogram_comparison, stock_endpoint, validate_run,
)
from profile_native import digest, wheel_identity, write_json

TOP = 12
FOCUS = 6
CALLERS = 4
EVENTS = 20_000


def symbol(entry):
    file, line, name = entry
    return {"file": file, "line": line, "name": name}


def counters(values):
    primitive, calls, self_s, cumulative_s = values
    return {"primitive_calls": primitive, "calls": calls, "self_s": self_s, "cumulative_s": cumulative_s}


def profile_record(stats, entry):
    *values, callers = stats[entry]
    hottest = sorted(callers.items(), key=lambda item: (-item[1][3], -item[1][2], item[0]))[:CALLERS]
    return {**symbol(entry), **counters(values), "distinct_direct_callers": len(callers),
            "callers": [{**symbol(caller), **counters(edge)} for caller, edge in hottest]}


def rank(entries, stats, metric, limit):
    return [profile_record(stats, key)
            for key in sorted(entries, key=lambda key: (-stats[key][metric], key))[:limit]]


def summarize(profile, visible):
    raw = pstats.Stats(str(profile))
    stats = raw.stats
    if not stats or raw.total_calls <= 0:
        raise ValueError("The Python profiling log has no calls")
    if not any(Path(file) == visible and name == "main" for file, _, name in stats):
        raise ValueError("cProfile did not record the shipped visible main function")
    focus = {
        "numpy_astype": [],
        "numpy_where": [],
        "array_preparation": [],
        "histogram": [],
        "nnjvt_orchestration": [],
        "visible_script": [],
    }
    for entry in stats:
        file, _, name = entry
        own = Path(file) == visible
        lower = name.lower()
        if ("numpy" in lower and "astype" in lower) or ("numpy" in file and lower == "astype"):
            focus["numpy_astype"].append(entry)
        if ("numpy" in lower and "where" in lower) or ("numpy" in file and lower == "where"):
            focus["numpy_where"].append(entry)
        if (file.endswith(("/atlas_jet_demo/inputs.py", "/atlas_jet_demo/transport.py")) or
                own and name in {"event_inputs", "truth_context", "uncertainty_inputs", "status_bits", "p4_fields"}):
            focus["array_preparation"].append(entry)
        if ("/hist/" in file or "/boost_histogram/" in file
                or own and name == "create_histogram" or "boost_histogram" in lower):
            focus["histogram"].append(entry)
        called_from_jvt_selection = any(
            Path(caller_file) == visible and caller_name == "small_jvt_selection"
            for caller_file, _, caller_name in stats[entry][4]
        )
        if own and (any(marker in lower for marker in ("nnjvt", "small_jvt"))
                    or name == "small_sf" or (name == "evaluate" and called_from_jvt_selection)):
            focus["nnjvt_orchestration"].append(entry)
        if own:
            focus["visible_script"].append(entry)
    return {
        "classification": "one deterministic cProfile diagnostic outside formal timing",
        "basis": "cProfile elapsed seconds while each function is active, including any called native work",
        "interpretation": "Cumulative times overlap. Caller edges are direct. The profile measures calls and time.",
        "total_self_s": raw.total_tt, "total_calls": raw.total_calls, "primitive_calls": raw.prim_calls,
        "functions": len(stats), "top_by_self": rank(stats, stats, 2, TOP), "top_by_cumulative": rank(stats, stats, 3, TOP),
        "focus": {name: {"matched_functions": len(entries), "by_self": rank(entries, stats, 2, FOCUS)}
                  for name, entries in focus.items()},
    }


def plain_text(summary):
    def display(row):
        location = f" ({row['file']}:{row['line']})" if row["file"] != "~" else ""
        return (f"{row['self_s']:9.6f} self s  {row['cumulative_s']:9.6f} cumulative s  "
                f"{row['calls']:9d} calls  {row['name']}{location}")

    def section(title, rows, lines):
        lines.extend(("", title))
        for row in rows:
            lines.append(display(row))
            for edge in row["callers"]:
                lines.append("  called by  " + display(edge))

    lines = ["Installed combined jet workflow through cProfile", summary["interpretation"],
             f"{summary['total_self_s']:.6f} profiled s across {summary['total_calls']} calls"]
    section("Largest self times", summary["top_by_self"], lines)
    section("Largest cumulative times", summary["top_by_cumulative"], lines)
    for name, group in summary["focus"].items():
        section(f"{name.replace('_', ' ')} ({group['matched_functions']} functions)", group["by_self"], lines)
    return "\n".join(lines) + "\n"


def main(python: Path, demo: Path, wheel: Path, original: Path, output: Path, stock: Path):
    """Profile one installed combined full pass using the script's delivered defaults."""
    sys.path.insert(0, str(ROOT / "tools/reference"))
    from benchmark import CPU, run_leg

    if os.sched_getaffinity(0) != {CPU} or len(list(Path("/proc/self/task").iterdir())) != 1:
        raise ValueError(f"Launch the cProfile controller with one thread on CPU {CPU}")
    python = python.absolute()  # Preserve the pristine venv name.
    demo, wheel, original, stock = (path.resolve(strict=True) for path in (demo, wheel, original, stock))
    output = output.resolve()
    with (ROOT / f"results/.benchmark-cpu{CPU}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = json.loads((ROOT / "config/jets-input.json").read_text())
        if (source["entry_interval"] != [0, EVENTS] or original != Path(source["path"]).resolve(strict=True)
                or original.stat().st_size != source["bytes"] or digest(original) != source["sha256"]):
            raise ValueError("Use the complete original PHYSLITE file frozen by the campaign")
        bundle = wheel.with_name("bundle.json")
        packaged, metadata = checked_installation(python, demo, wheel, bundle, digest, wheel_identity)
        authority = stock_endpoint(stock, "both", original, source, digest)
        output.mkdir()
        histogram = output / "histograms.npz"
        profile = output / "profile.pstats"
        argv = [str(python), "-m", "cProfile", "-o", str(profile),
                str(demo), str(original), str(histogram), "--events", str(EVENTS)]
        write_json(output / "command.json", argv)
        write_json(output / "provenance.json", {
            "scope": "both", "mode": "full", "events": EVENTS, "passes": 1, "cpu": CPU, "threads": 1,
            "measurement": "deterministic cProfile diagnostic, separate from retained timing",
            "original": source, "stock": authority, "selected_defaults": metadata["demo_defaults"],
            "demo": {"path": str(demo), "sha256": digest(demo)}, "wheel": packaged,
            "bundle": {"path": str(bundle), "sha256": digest(bundle)},
        })
        env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
               "PYTHONDONTWRITEBYTECODE": "1", "ATLAS_JET_LIBRARY_AUDIT": str(output / "library-maps.json"),
               **dict.fromkeys(EXECUTION_THREADS, "1")}
        process = run_leg(argv, env, output, "visible", event_source=original)
        if not process["source_mapping_observed"]:
            raise ValueError("The installed worker did not physically map the original input")
        evidence = validate_run(output, "both", EVENTS, source["tfile_uuid"], metadata, packaged)
        stock_histogram = Path(authority["histogram"]["path"])
        if digest(stock_histogram) != authority["histogram"]["sha256"]:
            raise ValueError("Stock histogram changed during the cProfile run")
        checked = histogram_comparison(stock_histogram, histogram, "both", EVENTS)
        write_json(output / "histogram-comparison.json", {"exact_fields": checked, "sha256": digest(histogram)})
        if not all(checked.values()):
            raise ValueError("Installed cProfile histogram differs from the independent stock histogram")
        summary = {**summarize(profile, demo), "passed": True, "source_and_runtime_audit": evidence,
                   "pstats": {"path": str(profile), "sha256": digest(profile), "bytes": profile.stat().st_size},
                   "exact_stock_histogram": True}
        write_json(output / "summary.json", summary)
        with (output / "summary.txt").open("x") as report:
            report.write(plain_text(summary))
        typer.echo(f"Profiled one exact {EVENTS}-event combined pass. Results in {output}.")


if __name__ == "__main__":
    typer.run(main)
