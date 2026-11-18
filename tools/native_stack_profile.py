"""Reduce perf's native samples to nonoverlapping tool attribution and routine tables."""

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess

import typer

from profile_native import write_json

HEADER = re.compile(r"^\s*(\d+)/(\d+)\s+\[(\d+)\]\s+(\d+)\.(\d+):\s+(\d+)\s+cpu-clock(?::u)?:\s*(.*)$")
STACK_LINE = re.compile(r"^\s*(?:0x)?[0-9a-fA-F]+\s+(.*)\s+\(([^()]*)\)(?:\s+\(inlined\))?\s*$")
UNKNOWN = ("[unknown]", "[unknown]")
STAGES = (
    ("joint_jes_jer_jms", "shared", ("JetUncertaintyModel::", "JetUncertaintyNode::")),
    ("eta_jes", "shared", ("EtaJES::",)),
    ("event_inputs", "shared", ("EventInputs::",)),
    ("pileup_truth_labelling", "shared", ("TruthLabels::",)),
    ("small_pileup", "small-R", ("PileupNode::", "SmallPileup::")),
    ("small_gsc", "small-R", ("GscNode::", "SmallGsc::", "SmallGSC::")),
    ("small_nnjvt_update", "small-R", ("NnJvtUpdate::",)),
    ("small_nnjvt_selection", "small-R", ("NnJvtSelect::",)),
    ("small_nnjvt_scale_factor", "small-R", ("NnJvtSf::",)),
    ("small_event_scale_factor", "small-R", ("EventSF::",)),
    ("large_jms", "large-R", ("LargeJmsNode::", "LargeJms::")),
    ("large_jmr_forward_fold", "large-R", ("LargeJmrNode::", "LargeJmr::")),
    ("large_selections", "large-R", ("LargeSelection::",)),
)
SCOPE = {stage: scope for stage, scope, _ in STAGES}


@dataclass
class Sample:
    pid: int
    tid: int
    cpu: int
    timestamp: int
    period: int
    frames: list


def valid_frames(raw):
    for line in raw:
        frame = STACK_LINE.fullmatch(line)
        if frame:
            symbol, dso = (value.strip() for value in frame.groups())
            if symbol.startswith("0x") or symbol == "[unknown]":
                symbol = "[unknown]"
            yield dso, symbol


def samples(lines, defects):
    current = None
    for line in lines:
        line = line.rstrip("\n")
        header = HEADER.fullmatch(line)
        if header:
            if current is not None:
                yield current
            pid, tid, cpu, sec, frac, period, inline = header.groups()
            timestamp = int(sec) * 1_000_000_000 + int(frac.ljust(9, "0"))
            current = Sample(int(pid), int(tid), int(cpu), timestamp, int(period), list(valid_frames([inline])))
        elif "PERF_RECORD_LOST" in line:
            defects["lost_events"] += 1
        elif line.strip() and current is not None:
            frame = list(valid_frames([line]))
            if frame:
                current.frames.extend(frame)
            else:
                defects["unparsed_stack_lines"] += 1
        elif line.strip():
            defects["unparsed_prefix_lines"] += 1
    if current is not None:
        yield current


def owner(frames):
    for dso, symbol in frames:
        if not dso.rsplit("/", 1)[-1].startswith("libjets.so"):
            continue
        for stage, _, anchors in STAGES:
            if any(anchor in symbol for anchor in anchors):
                return stage
    dsos = tuple(dso for dso, _ in frames)
    names = tuple(dso.rsplit("/", 1)[-1] for dso in dsos)
    if (any(name.startswith("libcorrectionlib.so") for name in names)
            or any("/correctionlib/_core.cpython" in dso for dso in dsos)):
        return "correctionlib_unattributed"
    if any(name.startswith("libjets.so") for name in names):
        return "provider_unattributed"
    if any("/boost_histogram/" in dso for dso in dsos):
        return "histogram_fill"
    if any(dso.startswith(("libz.so", "liblz", "libzstd", "cramjam")) for dso in names):
        return "decompression"
    if any(dso.startswith(("libawkward", "_ext.cpython", "_multiarray_umath")) for dso in names):
        return "array_operations"
    if names and all(name == "[unknown]" for name in names):
        return "unresolved"
    return "python_or_other_native"


def rows(counter, total):
    return [{"dso": dso, "symbol": symbol, "period_ns": period, "percent_of_window": 100.0 * period / total}
            for (dso, symbol), period in counter.most_common()]


def write_tsv(path, records):
    if not records:
        return
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, records[0], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def summarize_samples(lines, recording):
    timing = recording["measurement"]
    lower, higher = timing["start_ns"], timing["end_ns"]
    pid = recording["pid"]
    counts, defects, resolved_depth = Counter(), Counter(), Counter()
    by_stage, self_weights, inclusive_weights, stack_weights = Counter(), Counter(), Counter(), Counter()
    by_stage_self, by_stage_inclusive, by_stage_count = defaultdict(Counter), defaultdict(Counter), Counter()
    stage_traces = defaultdict(Counter)
    for sample in samples(lines, defects):
        if not lower <= sample.timestamp < higher:
            counts["out_of_processing_window"] += 1
            continue
        if sample.pid != pid or sample.tid != pid or sample.cpu != 3:
            counts["unexpected_cpu_or_worker"] += 1
            continue
        if sample.period <= 0:
            counts["zero_period"] += 1
            continue

        counts["processing"] += 1
        frames = sample.frames or [UNKNOWN]
        leaf = frames[0]
        stage = owner(frames)
        period = sample.period
        unique = set(frames)
        depth = sum(symbol != "[unknown]" for _, symbol in frames)
        resolved_depth[depth] += 1
        if leaf[1] == "[unknown]":
            counts["unresolved_leaf"] += 1
        self_weights[leaf] += period
        by_stage[stage] += period
        by_stage_count[stage] += 1
        by_stage_self[stage][leaf] += period
        for frame in unique:
            inclusive_weights[frame] += period
            by_stage_inclusive[stage][frame] += period
        stack_weights[tuple(frames)] += period
        # Limit anchors to the leaf and numerical callers; the original full
        # machine stacks remain available in stacks.txt and the folded file.
        if stage in SCOPE:
            trace = [leaf]
            trace.extend(frame for frame in frames[1:] if frame[0].rsplit("/", 1)[-1].startswith("libjets.so"))
            stage_traces[stage][tuple(dict.fromkeys(trace))] += period

    total = sum(by_stage.values())
    if not total:
        raise ValueError(f"No parser-verified perf samples in the measured processing window: {dict(defects)}")
    owners = [{
        "owner": stage, "scope": SCOPE.get(stage, "infrastructure/unknown"),
        "samples": by_stage_count[stage], "period_ns": weight, "percent_of_window": 100.0 * weight / total,
    } for stage, weight in by_stage.most_common()]
    stage_tables = {}
    for stage, stage_period in by_stage.items():
        stage_tables[stage] = {}
        for mode, table in (("self", by_stage_self), ("inclusive", by_stage_inclusive)):
            entries = rows(table[stage], total)
            for entry in entries:
                entry["percent_of_owner"] = 100.0 * entry["period_ns"] / stage_period
            stage_tables[stage][mode] = entries
    examples = {
        stage: [
            {"period_ns": weight, "leaf_to_provider_callers": [{"dso": dso, "symbol": symbol} for dso, symbol in stack]}
            for stack, weight in traces.most_common(10)
        ]
        for stage, traces in stage_traces.items()
    }
    return {
        "samples": dict(counts), "parse_defects": dict(defects), "resolved_frame_depth_histogram": dict(resolved_depth),
        "denominator_period_ns": total, "owners": owners, "self": rows(self_weights, total),
        "inclusive": rows(inclusive_weights, total), "within_owner": stage_tables, "concrete_provider_stacks": examples,
        "stack_weights": stack_weights,
    }


def summarize(recorded: Path, perf: Path, output: Path | None = None):
    recorded = recorded.resolve()
    if output is None:
        output = recorded
    else:
        output = output.resolve()
        output.mkdir(exist_ok=False)
    recording = json.loads((recorded / "recording.json").read_text())
    if not recording["passed"]:
        raise ValueError("Complete-pass validation is required before attributing samples")
    raw = Path(recording["profile"]["path"])
    if raw.stat().st_size != recording["profile"]["bytes"]:
        raise ValueError("Saved perf data size changed since recording")
    timing = recording["measurement"]
    window = f"{timing['start_ns'] // 10**9}.{timing['start_ns'] % 10**9:09d},"
    window += f"{timing['end_ns'] // 10**9}.{timing['end_ns'] % 10**9:09d}"
    base = [str(perf)]
    steps = [
        ("header.txt", ["report", "--header-only", "--stdio", "-i", str(raw)]),
        ("buildids.txt", ["buildid-list", "-i", str(raw)]),
        ("stacks.txt", ["script", "--ns", "--demangle", "--inline", "--show-lost-events", "--show-kernel-path",
                        "-F", "pid,tid,cpu,time,event,period,ip,sym,dso", "-i", str(raw)]),
    ]
    with (output / "postprocess.log").open("x", encoding="utf-8") as errors:
        for name, args in steps:
            with (output / name).open("x", encoding="utf-8") as stream:
                subprocess.run([*base, *args], stdout=stream, stderr=errors, check=True)
    with (output / "stacks.txt").open(encoding="utf-8", errors="replace") as stream:
        summary = summarize_samples(stream, recording)
    stacks = summary.pop("stack_weights")
    with (output / "stacks.folded").open("x", encoding="utf-8") as folded:
        for frames, period in stacks.most_common():
            stack = ";".join(f"{dso}!{symbol}".replace(";", ",") for dso, symbol in reversed(frames))
            folded.write(f"{stack} {period}\n")
    for name in ("self", "inclusive", "owners"):
        write_tsv(output / f"{name}.tsv", summary[name])
    for attribution in ("self", "inclusive"):
        results = [{"owner": owner, **entry} for owner, table in summary["within_owner"].items()
                   for entry in table[attribution]]
        write_tsv(output / f"within_owner_{attribution}.tsv", results)
    summary.update({
        "passed": bool(
            not summary["samples"].get("unexpected_cpu_or_worker") and
            not summary["parse_defects"].get("lost_events") and
            summary["concrete_provider_stacks"].get("joint_jes_jer_jms")
        ),
        "clock_window": window, "measurement": timing, "recording": str(recorded / "recording.json"),
        "units": "CPU-clock sample periods in ns, not unprofiled wall-time or per-tool elapsed time",
        "self_definition": "one leaf per sample; unresolved leaves still count in the denominator",
        "owner_definition": "nearest identifiable provider stage on the stack; one owner per sample",
        "inclusive_definition": "a routine counts once per sample even when recursive; routines overlap on the same stack",
        "artifact": recording["wheel"],
    })
    write_json(output / "summary.json", summary)
    typer.echo(f"{summary['samples']['processing']} processing-window samples in {output / 'summary.json'}")
    if not summary["passed"]:
        raise RuntimeError("The stack capture needs review for sample loss, CPU/worker affinity or missing uncertainty anchors")


def main(recorded: Path, perf: Path = Path("/usr/bin/perf"), analysis_output: Path | None = None):
    """Decode a saved recording without executing the calibration again."""
    try:
        summarize(recorded, perf, analysis_output)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        failure = (analysis_output or recorded) / "postprocess-failure.json"
        if failure.parent.is_dir() and not failure.exists():
            write_json(failure, {"passed": False, "error": str(error)})
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error


if __name__ == "__main__":
    typer.run(main)
