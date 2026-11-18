"""Collect full-file engineering pairs, their physics checks and guest-load observations."""
import hashlib
import json
import os
from pathlib import Path

import typer


def cpu_counters(snapshot):
    return {line.split()[0]: [int(value) for value in line.split()[1:]]
            for line in snapshot.get("all_guest_cpus", [])}


def busy(ticks):
    return sum(ticks[i] for i in (0, 1, 2, 5, 6, 7))


def main(trials: Path, output: Path):
    inputs = json.loads((trials/"inputs.json").read_text())
    summary = json.loads((trials/"summary.json").read_text())
    rows = []
    ticks = os.sysconf("SC_CLK_TCK")
    for file in sorted(trials.glob("*/receipt.json")):
        record = json.loads(file.read_text())
        a, b = cpu_counters(record["before"]), cpu_counters(record["after"])
        item = {key:value for key,value in record.items() if key not in ("before", "after")}
        item["receipt"] = str(file.resolve())
        item["receipt_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
        item["worker_cpu_over_wall"] = item["cpu_s"]/item["wall_s"]
        if a:
            reserved = busy(b["cpu3"]) - busy(a["cpu3"])
            global_busy = busy(b["cpu"]) - busy(a["cpu"])
            item["other_guest_core_equivalents"] = (global_busy - reserved)/(ticks*item["wall_s"])
            item["cpu3_iowait_s"] = (b["cpu3"][4]-a["cpu3"][4])/ticks
        rows.append(item)
    if len(rows)!=summary["observations"] or not all(row["exact_histogram"] for row in rows):
        raise ValueError("Incomplete results or failed original histograms")
    with output.open("x") as stream:
        json.dump({"plan":inputs, "summary":summary, "rows":rows}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    typer.run(main)
