"""Independent stock and direct endpoint distributions from a comparison pass."""

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path

import awkward as ak
import numpy as np

from columnar.inputs import flatten


@dataclass(frozen=True)
class Observable:
    branch: str
    unit: str
    edges: tuple
    divisor: int = 1


PT = (0, 20, 30, 40, 60, 80, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000)
MASS = (-100, -40, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 30, 40, 60, 80,
        100, 150, 200, 300, 400, 500, 600, 750, 1000)
SCALE_FACTOR = (-2, -1.001, -0.999, *np.linspace(0, 2, 81), 3, 5, 10)
CORE = {
    "pt": Observable("p4pt", "GeV", PT, 1000),
    "eta": Observable("p4eta", "pseudorapidity", tuple(np.linspace(-5, 5, 101))),
    "phi": Observable("p4phi", "radian", tuple(np.linspace(-math.pi, math.pi, 81))),
    "mass": Observable("p4m", "GeV", MASS, 1000),
}
SMALL = {
    **CORE,
    "nnjvt_score": Observable("nnjvt", "score", (-2, -1.001, -0.999, *np.linspace(0, 1, 51), 1.001, 2)),
    "nnjvt_sf": Observable("jvtEfficiency", "dimensionless", SCALE_FACTOR),
    "histogram_weight": Observable("hist_weight", "dimensionless", SCALE_FACTOR),
    "event_sf": Observable("eventSF", "dimensionless", SCALE_FACTOR),
}
TALLIES = {
    "sr": ("jesJerValidity", "jvtValidity", "nnjvtPass", "jvtSelection", "isJvtHS"),
    "lr": ("jesJERJMSValidity", "jmrValidity", "selectPtEta", "selectMass"),
}
KNOWN_STATUS = {"0": "absent", "4294967295": "valid/accept", "4294967294": "out of validity"}
RESIDUAL = (-1.0, -1e-1, -1e-2, -1e-3, -1e-4, -1e-5, -1e-6, -1e-7,
            -1e-8, -1e-9, 0.0, 1e-9, 1e-8, 1e-7, 1e-6,
            1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def bin_counts(values, edges):
    indices = np.searchsorted(edges, values[~np.isnan(values)], side="right")
    return np.bincount(indices, minlength=len(edges) + 1)


def counts_by_code(values):
    keys, counts = np.unique(values, return_counts=True)
    return dict(zip(map(int, keys), map(int, counts), strict=True))


def representative_bins(counts, edges):
    total = int(counts.sum())
    if not total:
        return {}
    cumulative = counts.cumsum()
    representatives = {}
    for name, quantile in (("low", 0.1), ("median", 0.5), ("high", 0.9)):
        bin_index = int(np.searchsorted(cumulative, math.ceil(quantile * total)))
        representatives[name] = {
            "quantile": quantile,
            "bin_index": bin_index,
            "low_edge": None if bin_index == 0 else edges[bin_index - 1],
            "high_edge": None if bin_index == len(edges) else edges[bin_index],
            "entries_in_bin": int(counts[bin_index]),
        }
    return representatives


class Distribution:
    def __init__(self, observable, states):
        self.observable = observable
        self.edges = np.asarray(observable.edges, dtype=np.float64)
        shape = (len(states), len(self.edges) + 1)
        self.stock = np.zeros(shape, dtype=np.int64)
        self.direct = np.zeros(shape, dtype=np.int64)
        self.residual = np.zeros((len(states), len(RESIDUAL) + 1), dtype=np.int64)
        self.stock_nan = np.zeros(len(states), dtype=np.int64)
        self.direct_nan = np.zeros(len(states), dtype=np.int64)
        self.paired = np.zeros(len(states), dtype=np.int64)
        self.unpaired_chunks = np.zeros(len(states), dtype=np.int64)
        self.exact_mismatches = np.zeros(len(states), dtype=np.int64)
        self.max_abs_residual = np.zeros(len(states), dtype=np.float64)

    def fill(self, index, stock, direct, same_event_counts):
        a = np.asarray(stock, dtype=np.float64) / self.observable.divisor
        c = np.asarray(direct, dtype=np.float64) / self.observable.divisor
        self.stock[index] += bin_counts(a, self.edges)
        self.direct[index] += bin_counts(c, self.edges)
        self.stock_nan[index] += np.isnan(a).sum()
        self.direct_nan[index] += np.isnan(c).sum()
        if not same_event_counts or a.shape != c.shape:
            self.unpaired_chunks[index] += 1
            return
        self.paired[index] += a.size
        self.exact_mismatches[index] += np.count_nonzero((stock != direct) & ~(np.isnan(a) & np.isnan(c)))
        finite = np.isfinite(a) & np.isfinite(c)
        delta = c[finite] - a[finite]
        self.residual[index] += bin_counts(delta, RESIDUAL)
        self.max_abs_residual[index] = max(self.max_abs_residual[index], np.max(np.abs(delta), initial=0.0))

    def persist(self, key, arrays, states):
        arrays[f"{key}_edges"] = self.edges
        arrays[f"{key}_stock_counts"] = self.stock
        arrays[f"{key}_direct_counts"] = self.direct
        arrays[f"{key}_count_residual"] = self.direct - self.stock
        arrays[f"{key}_pair_residual_counts"] = self.residual
        arrays[f"{key}_stock_nan"] = self.stock_nan
        arrays[f"{key}_direct_nan"] = self.direct_nan
        arrays[f"{key}_pairs"] = self.paired
        arrays[f"{key}_unpaired_chunks"] = self.unpaired_chunks
        arrays[f"{key}_exact_mismatches"] = self.exact_mismatches
        arrays[f"{key}_max_abs_residual"] = self.max_abs_residual
        absolute_change = np.abs(self.stock[1:] - self.stock[0]).sum(axis=1)
        ascending = np.argsort(absolute_change, kind="stable")
        examples = {}
        if ascending.size:
            for label, rank in (("low", 0), ("median", (ascending.size - 1) // 2), ("high", ascending.size - 1)):
                offset = ascending[rank]
                examples[label] = {"state": states[offset + 1], "stock_l1_bin_count_change": int(absolute_change[offset])}
        return {
            "unit": self.observable.unit,
            "edges": self.edges.tolist(),
            "stock_and_direct_bin_mismatches_per_state": np.count_nonzero(self.stock != self.direct, axis=1).tolist(),
            "representative_shift_states": examples,
            "plot_nominal_quantiles": {
                "stock": representative_bins(self.stock[0], self.edges.tolist()),
                "direct": representative_bins(self.direct[0], self.edges.tolist()),
            },
        }


class EndpointDistributions:
    def __init__(self, states, weight_only, valid):
        self.states = states
        self.positions = {prefix: {name: i for i, name in enumerate(names)} for prefix, names in states.items()}
        self.observables = {p: SMALL if p == "sr" else CORE for p in states}
        self.distributions = {p: {key: Distribution(obs, states[p]) for key, obs in variables.items()}
                              for p, variables in self.observables.items()}
        self.tallies = {p: {name: {key: (Counter(), Counter()) for key in TALLIES[p]} for name in names}
                        for p, names in states.items()}
        self.nominal = {}
        self.stock_arrays = {}
        self.weight_only = weight_only
        self.valid = valid

    def begin_chunk(self):
        self.nominal.clear()
        self.stock_arrays.clear()

    def stock(self, reference, key, event_scalar=False):
        if key not in self.stock_arrays:
            value = reference[key]
            self.stock_arrays[key] = np.asarray(ak.to_numpy(value)) if event_scalar else flatten(value)
        return self.stock_arrays[key]

    def record(self, frame, reference):
        prefix, state = frame.prefix, frame.state
        weight_state = prefix == "sr" and state in self.weight_only
        if state == "NOSYS":
            self.nominal[prefix] = frame.fields
        kin = "NOSYS" if weight_state else state
        kin_fields = self.nominal[prefix] if weight_state else frame.fields
        index = self.positions[prefix][state]
        ref_pt = reference[f"{prefix}_p4pt_{kin}"]
        same_jet_counts = np.array_equal(ak.to_numpy(ak.num(ref_pt, axis=1)), frame.counts)
        for key, observable in self.observables[prefix].items():
            field = observable.branch
            if field == "hist_weight":
                raw = self.stock(reference, f"sr_jvtEfficiency_{state}")
                valid = self.stock(reference, f"sr_jvtValidity_{kin}")
                stock = np.where(valid == self.valid, raw, 1.0)
            elif field in ("eventSF", "jvtEfficiency"):
                stock = self.stock(reference, f"{prefix}_{field}_{state}", event_scalar=field == "eventSF")
            else:
                stock = self.stock(reference, f"{prefix}_{field}_{kin}")
            direct = frame.fields[field] if field in ("hist_weight", "eventSF", "jvtEfficiency") else kin_fields[field]
            self.distributions[prefix][key].fill(index, stock, direct, field == "eventSF" or same_jet_counts)

        for key, (stock_counts, direct_counts) in self.tallies[prefix][state].items():
            stock = self.stock(reference, f"{prefix}_{key}_{kin}")
            stock_counts.update(counts_by_code(stock))
            direct_counts.update(counts_by_code(kin_fields[key]))

    def write(self, directory: Path, report_path: Path, entries, ff_names):
        directory.mkdir()
        arrays = {"entries": np.asarray(entries), "paired_residual_edges": np.asarray(RESIDUAL)}
        collections = {}
        for prefix, states in self.states.items():
            arrays[f"{prefix}_states"] = np.asarray(states)
            observables = {field: distribution.persist(f"{prefix}_{field}", arrays, states)
                           for field, distribution in self.distributions[prefix].items()}
            collections[prefix] = {
                "observables": observables,
                "states": [
                    {
                        "name": state,
                        "position": index,
                        "kinematics": "shared_nominal" if state in self.weight_only else "configured",
                        "large_jmr": ("evaluated" if state in ff_names else "structurally_absent") if prefix == "lr" else None,
                        "codes": {
                            key: {"stock": dict(a), "direct": dict(c), "exact_count_match": a == c}
                            for key, (a, c) in self.tallies[prefix][state].items()
                        },
                    }
                    for index, state in enumerate(states)
                ],
            }
        with (directory / "distributions.npz").open("xb") as output:
            np.savez(output, **arrays)
        with (directory / "metadata.json").open("x", encoding="utf-8") as output:
            json.dump({
                "entries": entries,
                "comparison": str(report_path.resolve()),
                "axes": {"counts": ["state", "bin"], "state_axis": "per-collection manifest order"},
                "bin_convention": "underflow, half-open [edge[i], edge[i+1]) bins, overflow; NaN counted separately",
                "paired_residual": "direct minus stock in the displayed unit for finite pairs in event/object order",
                "representative_shift_states": "low, lower median and high L1 bin-count changes relative to stock NOSYS; includes flows and uses manifest order to break ties",
                "input_population": "all original jets before final selections; event SF has one value per event",
                "effective_weight": "raw per-jet SF when validity accepts, otherwise 1",
                "correction_status_codes": KNOWN_STATUS,
                "selection_codes": "small NNJvt booleans and large unsigned selection masks; all-one mask accepts",
                "collections": collections,
            }, output, indent=2, allow_nan=False)
            output.write("\n")
