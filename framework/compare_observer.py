"""Compare the stock and overlay observers without using converted C output."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import awkward as ak
import numpy as np
import typer
import uproot

from compare_nominal import EVENT_KEYS, require, scalar


app = typer.Typer(add_completion=False)
JVT_FIELDS = (
    "jvtEfficiency", "eventSF", "jvtSelection", "jvtValidity", "nnjvt", "nnjvtPass", "isJvtHS",
)


def fields(tree, jvt_only):
    if not jvt_only:
        return set(tree.keys())
    selected = tuple(f"sr_{field}_" for field in JVT_FIELDS) + ("weight_jvt_effSF_",)
    return {name for name in tree.keys() if name.startswith(selected)}


def values(array):
    return ak.to_numpy(ak.flatten(array)) if array.ndim > 1 else ak.to_numpy(array)


def byte_digest(array):
    raw = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    return hashlib.sha256(raw.view(np.uint8)).hexdigest()


def ordered_bits(float_array):
    bits = float_array.view(np.uint32)
    return np.where(bits & 0x80000000, ~bits, bits | 0x80000000).astype(np.int64)


def difference(branch, stock, overlay, run, event):
    require(stock.ndim == overlay.ndim, f"rank mismatch in {branch}")
    lengths = ak.to_numpy(ak.num(stock)) if stock.ndim > 1 else None
    if lengths is not None:
        require(np.array_equal(lengths, ak.to_numpy(ak.num(overlay))), f"shape mismatch in {branch}")
    a, b = values(stock), values(overlay)
    require(a.dtype == b.dtype, f"dtype mismatch in {branch}: {a.dtype} and {b.dtype}")
    floats = np.issubdtype(a.dtype, np.floating)
    if floats:
        require(a.dtype == np.float32, f"unexpected float dtype in {branch}")
        bad = np.flatnonzero(a.view(np.uint32) != b.view(np.uint32))
    else:
        bad = np.flatnonzero(a != b)
    proof = {
        "dtype": str(a.dtype), "items": int(a.size), "bytes": int(a.nbytes),
        "stock_values_sha256": byte_digest(a), "framework_values_sha256": byte_digest(b),
    }
    cumulative = np.cumsum(lengths, dtype=np.int64) if lengths is not None else None
    if cumulative is not None:
        offsets = np.concatenate((np.zeros(1, dtype=np.int64), cumulative))
        proof["common_offsets_sha256"] = byte_digest(offsets)
    if not len(bad):
        return None, proof
    result = {"different": len(bad), "objects": len(a), "dtype": str(a.dtype), "examples": []}
    if floats:
        finite = np.isfinite(a) & np.isfinite(b)
        abs_diff = np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64))
        ulps = np.abs(ordered_bits(a[finite]) - ordered_bits(b[finite]))
        result["max_abs_difference"] = float(np.max(abs_diff)) if abs_diff.size else None
        result["max_ulp_difference"] = int(np.max(ulps)) if ulps.size else None
        result["different_nonfinite_class"] = int(np.count_nonzero(np.isfinite(a) != np.isfinite(b)))
    for index in bad[:5]:
        row = int(np.searchsorted(cumulative, index, side="right")) if cumulative is not None else int(index)
        item = {
            "entry": row, "run": int(run[row]), "event": int(event[row]),
            "reference": scalar(a[index]) if floats else int(a[index]),
            "framework": scalar(b[index]) if floats else int(b[index]),
        }
        if cumulative is not None:
            item["jet_index"] = int(index - (cumulative[row - 1] if row else 0))
        result["examples"].append(item)
    return result, proof


@app.command()
def compare(reference: Path, framework: Path, output: Path, jvt_only: bool = False,
            nominal: bool = False, require_full: bool = False,
            stock_histogram: Path | None = None, framework_histogram: Path | None = None) -> None:
    """Check the independent observer identities, states, widths, values and decisions."""
    with uproot.open(reference) as a_file, uproot.open(framework) as b_file:
        a_tree, b_tree = a_file["analysis"], b_file["analysis"]
        events = b_tree.num_entries
        require(0 < events <= a_tree.num_entries, "invalid framework interval")
        if require_full:
            require(events == a_tree.num_entries, "framework stopped before the stock observer endpoint")
        a_states = a_file["systematicsList"].axis().labels()
        b_states = b_file["systematicsList"].axis().labels()
        if nominal:
            require(b_states == ["NOSYS"], f"unexpected nominal service states: {b_states}")
        else:
            require(a_states == b_states, "service systematic enumeration differs from stock")
        a_weights = a_file["systematicsListOnlyWeights"].axis().labels()
        b_weights = b_file["systematicsListOnlyWeights"].axis().labels()
        if nominal:
            require(not b_weights or b_weights == [""], f"unexpected nominal weight states: {b_weights}")
        else:
            require(a_weights == b_weights, "weight-only systematic enumeration differs from stock")
        a_fields, b_fields = fields(a_tree, jvt_only), fields(b_tree, jvt_only)
        if nominal:
            variations = tuple(f"_{state}" for state in a_states if state != "NOSYS")
            a_fields = {key for key in a_fields if not key.endswith(variations)}
        require(a_fields == b_fields, f"observer branches differ: {sorted(a_fields ^ b_fields)[:12]}")
        identity = list(EVENT_KEYS) + ["sr_index_NOSYS", "lr_index_NOSYS"]
        a_identity = a_tree.arrays(identity, entry_stop=events, how=dict)
        b_identity = b_tree.arrays(identity, how=dict)
        for key in identity:
            require(ak.almost_equal(a_identity[key], b_identity[key], rtol=0, atol=0),
                    f"observer identities differ for {key}")

        proofs = {}
        differences = {}
        names = sorted(b_fields)
        for start in range(0, len(names), 128):
            packet = names[start:start + 128]
            a = a_tree.arrays(packet, entry_stop=events, how=dict)
            b = b_tree.arrays(packet, how=dict)
            for branch in packet:
                require(a_tree[branch].typename == b_tree[branch].typename,
                        f"persisted ROOT type differs in {branch}")
                result, proof = difference(
                    branch, a[branch], b[branch], a_identity["runNumber"], a_identity["eventNumber"],
                )
                proofs[branch] = proof
                if result is not None:
                    differences[branch] = result

    require(bool(stock_histogram) == bool(framework_histogram), "provide both histogram files")
    histogram_proof = {}
    if stock_histogram and framework_histogram:
        with np.load(stock_histogram) as stock, np.load(framework_histogram) as candidate:
            require(set(stock.files) == set(candidate.files), "histogram field inventory differs")
            for key in sorted(stock.files):
                a, b = stock[key], candidate[key]
                require(a.dtype == b.dtype and a.shape == b.shape, f"histogram shape or type differs in {key}")
                hashes = byte_digest(a), byte_digest(b)
                histogram_proof[key] = {
                    "dtype": str(a.dtype), "shape": a.shape, "bytes": int(a.nbytes),
                    "stock_values_sha256": hashes[0], "framework_values_sha256": hashes[1],
                    "exact": hashes[0] == hashes[1],
                }
        require(bool(histogram_proof), "empty histogram file")
    exact_histograms = all(proof["exact"] for proof in histogram_proof.values())
    prefixes = Counter(
        branch.split("_", 1)[0] if "_" in branch else "event_identity" for branch in proofs
    )
    with output.open("x") as stream:
        json.dump({
            "reference": str(reference), "framework": str(framework),
            "events": events, "service_states": len(b_states), "service_state_labels": b_states,
            "weight_only_states": b_weights,
            "histograms": histogram_proof, "exact_histogram_parity": exact_histograms if histogram_proof else None,
            "branches": len(names), "branch_groups": prefixes,
            "checked_values": sum(proof["items"] for proof in proofs.values()),
            "checked_value_bytes": sum(proof["bytes"] for proof in proofs.values()),
            "different_branches": len(differences),
            "exact_observer_parity": not differences, "differences": differences, "proofs": proofs,
        }, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if differences or not exact_histograms:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
