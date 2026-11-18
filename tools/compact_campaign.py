"""Collect accepted attempts from one campaign without including multi-GB event files."""
import hashlib
import json
import os
from pathlib import Path

import typer


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(campaign: Path, output: Path):
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
    with output.open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    typer.run(main)
