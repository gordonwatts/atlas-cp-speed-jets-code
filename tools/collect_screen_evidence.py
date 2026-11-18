"""Collect single-change jet screening results without duplicating full per-object archives."""
from hashlib import file_digest
import json
from pathlib import Path
from typing import Annotated

import typer


def digest(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def main(output: Path, trial: Annotated[list[Path], typer.Option()]):
    summaries = []
    for case in trial:
        case = case.resolve(strict=True)
        work = case / "workflow"
        summary = json.loads((work / "summary.json").read_text())
        origins = json.loads((work / "inputs.json").read_text())
        objects = json.loads((case / "object-screen.json").read_text())
        comparisons = []
        for check in objects["cases"]:
            proof = Path(check["proof"])
            data = json.loads(proof.read_text())
            if not data["passed"]:
                raise ValueError(f"Independent object check failed: {proof}")
            comparisons.append({"arm": check["variant"], "scope": check["scope"],
                                "entries": data["entries_compared"], "path": str(proof), "sha256": digest(proof)})
        observations = []
        for path in sorted(work.glob("*/receipt.json")):
            row = json.loads(path.read_text())
            if not (row["exit"] == 0 and row["exact_histogram"] and row["process"]["passed"]
                    and row["matching_source_branches"]):
                raise ValueError(f"Full original-data trial failed: {path}")
            observations.append({key: row[key] for key in
                                 ("scope", "name", "round", "wall_s", "cpu_s", "peak_rss_kB",
                                  "source_branch_sha256", "exact_histogram")})
            observations[-1].update(receipt=str(path), receipt_sha256=digest(path))
        if len(observations) != summary["observations"] or not objects["passed"]:
            raise ValueError(f"Incomplete trial: {case}")
        summaries.append({"family": origins["plan"]["experiment"], "root": str(case), "inputs": origins,
                          "stock_object_gates": comparisons, "full_file_jobs": observations,
                          "metrics": summary["metrics"]})
    with output.open("x") as stream:
        json.dump({"cases": summaries}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    typer.run(main)
