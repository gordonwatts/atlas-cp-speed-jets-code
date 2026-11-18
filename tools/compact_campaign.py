"""Collect accepted attempts from one campaign without including multi-GB event files."""
import hashlib
import json
import os
from pathlib import Path
from typing import Annotated

import typer


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def guest_activity(leg, intervals):
    if "start_unix_ns" not in leg:
        return None
    begin, end = leg["start_unix_ns"], leg["end_unix_ns"]
    totals = {"jets": 0.0, "muon": 0.0, "other": 0.0, "exited": 0.0}
    covered = 0
    for row in intervals:
        first, last = row["begin_unix_ns"], row["end_unix_ns"]
        overlap = min(end, last) - max(begin, first)
        if overlap <= 0:
            continue
        covered += overlap
        for process in row["active_processes"]:
            totals[process["project"]] += process["cpu_s"] * overlap / (last - first)
    return {"start_unix_ns": begin, "end_unix_ns": end, "coverage_fraction": covered / (end - begin),
            "sampled_process_cpu_s": totals,
            "external_core_equivalents": sum(t for project, t in totals.items() if project != "jets") / leg["wall_s"]}


def main(campaign: Path, output: Path, guest_log: Annotated[list[Path] | None, typer.Option()] = None):
    intervals = []
    for path in guest_log or ():
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["type"] == "interval":
                    intervals.append(row)
    intervals.sort(key=lambda row: row["begin_unix_ns"])
    if any(a["end_unix_ns"] > b["begin_unix_ns"] for a, b in zip(intervals, intervals[1:])):
        raise ValueError("Guest monitors cover overlapping time intervals")

    summary = json.loads((campaign / "summary.json").read_text())
    provenance = json.loads((campaign / "campaign.json").read_text())
    rows = []
    ticks = os.sysconf("SC_CLK_TCK")
    for job_name, directory in sorted(summary["progress"].items()):
        attempt = Path(directory)
        receipt = json.loads((attempt / "receipt.json").read_text())
        row = {"job": job_name, "attempt": str(attempt.resolve()), **receipt}
        legs = {}
        for filename, expected in receipt["resource_receipts_sha256"].items():
            path = attempt / filename
            if digest(path) != expected:
                raise ValueError(f"Modified resource receipt {path}")
            data = json.loads(path.read_text())
            name = path.stem
            deltas = receipt["cpu_accounting"][name]["cpu_stat_delta"]
            legs[name] = {key: data[key] for key in ("wall_s","cpu_user_s","cpu_sys_s","peak_rss_kb","cpus_seen","threads_seen")}
            if intervals:
                legs[name]["guest_activity"] = guest_activity(data, intervals)
            legs[name].update(cpu3_iowait_s=deltas[4] / ticks,
                              cpu3_idle_s=deltas[3] / ticks,
                              other_cpu3_busy_s=receipt["cpu_accounting"][name]["unattributed_busy_s"])
        row["legs"] = legs
        ntuple = attempt / "reference.root"
        row["ntuple_disk_bytes"] = ntuple.stat().st_size if ntuple.exists() else None
        if digest(attempt / "histograms.npz") != receipt["histogram_sha256"]:
            raise ValueError(f"Modified histogram {attempt}")
        rows.append(row)
    payload = {"campaign": str(campaign.resolve()), "metadata": provenance, "summary": summary,
               "clock_ticks_per_second": ticks, "observations": rows}
    if guest_log:
        payload["guest_monitor"] = {
            "sources": {str(p.resolve()): digest(p) for p in guest_log},
            "method": "Intersects each worker's Unix-clock bounds with guest snapshots. CPU seconds within a partial interval are prorated. Processes must survive two snapshots and accumulate at least 0.05 CPU seconds. Averages are descriptive lower estimates.",
        }
    with output.open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    typer.run(main)
