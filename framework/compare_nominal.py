"""Validate the framework nominal stages against the independent stock observer."""
import json
import math
from pathlib import Path

import awkward as ak
import numpy as np
import typer
import uproot


app = typer.Typer(add_completion=False)
EVENT_KEYS = ("eventNumber", "runNumber", "lumiBlock", "mcChannelNumber")
COLLECTIONS = {
    "sr": ("pileup", "etaJES", "gsc", "calib"),
    "lr": ("etaJES", "jms", "calib"),
}
COORDINATES = ("pt", "eta", "phi", "m")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scalar(value):
    return float(value) if math.isfinite(value) else str(value)


def compare_branch(branch, ref, cand, run, event):
    lengths = ak.to_numpy(ak.num(ref))
    require(np.array_equal(lengths, ak.to_numpy(ak.num(cand))), f"jagged shape changed for {branch}")
    a = ak.to_numpy(ak.flatten(ref))
    b = ak.to_numpy(ak.flatten(cand))
    require(a.dtype == np.float32 and b.dtype == np.float32, f"{branch} has a non-float auxiliary type")
    different = np.flatnonzero(a.view(np.uint32) != b.view(np.uint32))
    finite = np.isfinite(a) & np.isfinite(b)
    delta = np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64))
    cumulative = np.cumsum(lengths)
    mismatches = []
    for index in different[:5]:
        item = int(np.searchsorted(cumulative, index, side="right"))
        offset = cumulative[item - 1] if item else 0
        mismatches.append({
            "entry": item,
            "run": int(run[item]),
            "event": int(event[item]),
            "jet_index": int(index - offset),
            "reference": scalar(a[index]),
            "framework": scalar(b[index]),
        })
    return {
        "objects": len(a),
        "different_float_bits": len(different),
        "different_nonfinite_class": int(np.count_nonzero(np.isfinite(a) != np.isfinite(b))),
        "max_abs_difference": float(np.max(delta)) if delta.size else None,
        "first_disagreements": mismatches,
    }


@app.command()
def compare(reference: Path, framework: Path, result: Path) -> None:
    """Check the five nominal nodes through the common AnalysisBase observer."""
    fields = list(EVENT_KEYS) + [f"{p}_index_NOSYS" for p in COLLECTIONS]
    branches = [
        f"{prefix}_{scale}{coord}_NOSYS"
        for prefix, scales in COLLECTIONS.items()
        for scale in scales
        for coord in COORDINATES
    ]
    with uproot.open(reference) as a_file, uproot.open(framework) as b_file:
        a_tree = a_file["analysis"]
        b_tree = b_file["analysis"]
        n = b_tree.num_entries
        require(0 < n <= a_tree.num_entries, "framework entries exceed the stock observer interval")
        a = a_tree.arrays(fields + branches, entry_stop=n, how=dict)
        b = b_tree.arrays(fields + branches, how=dict)
    for field in fields:
        require(ak.almost_equal(a[field], b[field], rtol=0, atol=0), f"identity changed for {field}")
    comparisons = {
        branch: compare_branch(branch, a[branch], b[branch], a["runNumber"], a["eventNumber"])
        for branch in branches
    }
    parity = all(row["different_float_bits"] == 0 for row in comparisons.values())
    with result.open("x") as stream:
        json.dump({
            "reference": str(reference), "framework": str(framework), "events": n,
            "exact_float_stage_parity": parity, "branches": comparisons,
        }, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if not parity:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
