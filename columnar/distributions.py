"""Independent stock and direct endpoint distributions from a comparison pass."""

from collections import Counter
from ctypes import CDLL, c_size_t
from dataclasses import dataclass
from hashlib import sha256
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


class Energy:
    def __init__(self, path):
        self.library = CDLL(str(path.resolve(strict=True)))
        self.evaluate = self.library.columnar_alias_energy
        self.evaluate.restype = None
        f32 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
        self.evaluate.argtypes = [c_size_t, f32, f32, f32, f32, f32]

    def __call__(self, p4):
        coordinates = [np.ascontiguousarray(values, dtype=np.float32) for values in p4]
        if len(coordinates) != 4 or any(values.shape != coordinates[0].shape for values in coordinates):
            raise ValueError("Energy requires four aligned coordinate vectors")
        output = np.empty_like(coordinates[0])
        self.evaluate(len(output), *coordinates, output)
        return output


class EnergyDistributions:
    def __init__(self, states):
        if not states:
            raise ValueError("Energy spectra require a calibrated small-R observer comparison")
        edges = (0, 20, 30, 40, 60, 80, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000,
                 3000, 4500, 7000, 10000)
        self.states = states
        self.index = {f"sr_e_{name}": i for i, name in enumerate(states)}
        self.distribution = Distribution(Observable("e", "GeV", edges, 1000), states)
        self.ranges = np.broadcast_to([np.inf, -np.inf], (len(states), 2, 2)).copy()

    def fill(self, name, stock, direct, same_counts):
        index = self.index[name]
        self.distribution.fill(index, stock, direct, same_counts)
        for side, values in enumerate((stock, direct)):
            finite = values[np.isfinite(values)]
            if finite.size:
                current = self.ranges[index, side]
                current[0] = min(current[0], float(finite.min()) / 1000)
                current[1] = max(current[1], float(finite.max()) / 1000)

    def persist(self, arrays):
        arrays["sr_energy_states"] = np.asarray(self.states)
        result = self.distribution.persist("sr_energy", arrays, self.states)
        result["state_axis"] = "sr_energy_states"
        arrays["sr_energy_stock_finite_range"] = self.ranges[:, 0]
        arrays["sr_energy_direct_finite_range"] = self.ranges[:, 1]
        return result

    def comparison(self):
        counter = self.distribution
        a_count = counter.stock.sum(axis=1) + counter.stock_nan
        c_count = counter.direct.sum(axis=1) + counter.direct_nan
        bins = int(np.count_nonzero(counter.stock != counter.direct))
        exact_mismatches = int(counter.exact_mismatches.sum())
        unpaired_chunks = int(counter.unpaired_chunks.sum())
        passed = (
            np.array_equal(a_count, c_count)
            and not (bins or exact_mismatches or unpaired_chunks)
            and np.array_equal(counter.stock_nan, counter.direct_nan)
            and np.array_equal(self.ranges[:, 0], self.ranges[:, 1])
        )
        return {
            "passed": passed, "states": len(self.states),
            "stock_values": int(a_count.sum()), "direct_values": int(c_count.sum()),
            "different_histogram_bins": bins, "unpaired_chunks": unpaired_chunks,
            "exact_mismatches": exact_mismatches,
            "max_abs_residual": float(counter.max_abs_residual.max(initial=0.0)),
        }

    def write(self, output, entries, sources, absence, proof):
        output.mkdir()
        arrays = {"entries": np.asarray(entries), "sr_states": np.asarray(self.states),
                  "paired_residual_edges": np.asarray(RESIDUAL)}
        metadata = self.persist(arrays)
        summary = self.comparison()
        with (output / "distributions.npz").open("xb") as stream:
            np.savez(stream, **arrays)
        with (output / "metadata.json").open("x", encoding="utf-8") as stream:
            json.dump({
                "passed": summary["passed"],
                "entries": entries,
                "kinematic_states": self.states,
                "histogram": metadata,
                "axes": {"counts": ["state", "bin"], "finite_range": ["state", "minimum_then_maximum"]},
                "bin_convention": "underflow, half-open [edge[i], edge[i+1]) bins, overflow; NaN counted separately",
                "authority_and_candidate": (
                    "A uses the persisted float energy. C reconstructs float energy using the native operation "
                    "on the already-proven exact per-object C four-vector representation."
                ),
                "proofs": sources,
                "observer_alias_proof": str(proof.resolve()),
                "stock_availability": absence,
            }, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return {"path": str(output.resolve()), **summary}


def observer_availability(keys):
    return {
        "large_r_energy": {"persisted": any(name == "lr_e" or name.startswith("lr_e_") for name in keys),
                           "alternative": "per-object pt, eta, phi and mass are persisted and verified"},
        "large_r_unmodified_auxiliary_branches": {
            field: sum(name.startswith(f"lr_{field}_") for name in keys)
            for field in ("d2", "tau1", "tau2", "tau3", "split12", "split23", "truthLabel")
        },
    }


class EndpointDistributions:
    def __init__(self, states, weight_only, valid, energy_library=None):
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
        self.energy = self.energy_distributions = None
        self.energy_identity = None
        self.stock_availability = None
        if energy_library is not None:
            self.energy = Energy(energy_library)
            self.energy_distributions = EnergyDistributions(
                [name for name in states.get("sr", []) if name not in weight_only])
            self.energy_identity = {
                "path": str(energy_library.resolve()),
                "sha256": sha256(energy_library.read_bytes()).hexdigest(),
                "candidate_inputs": "own evaluated C frame float pT, eta, phi, mass",
                "stock_inputs": "persisted sr_e systematic branches",
            }

    def observer_branches(self, stock_keys):
        self.stock_availability = observer_availability(stock_keys)
        return set(self.energy_distributions.index) if self.energy_distributions is not None else set()

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

        if self.energy is not None and prefix == "sr" and not weight_state:
            energy_key = f"sr_e_{state}"
            raw_energy = reference[energy_key]
            matches = np.array_equal(ak.to_numpy(ak.num(raw_energy, axis=1)), frame.counts)
            derived = self.energy(frame.fields[field] for field in ("p4pt", "p4eta", "p4phi", "p4m"))
            self.energy_distributions.fill(energy_key, flatten(raw_energy), derived, matches)

        for key, (stock_counts, direct_counts) in self.tallies[prefix][state].items():
            stock = self.stock(reference, f"{prefix}_{key}_{kin}")
            stock_counts.update(counts_by_code(stock))
            direct_counts.update(counts_by_code(kin_fields[key]))

    def comparison(self):
        summary = {}
        for prefix, distributions in self.distributions.items():
            for name, counter in distributions.items():
                summary[f"{prefix}_{name}"] = {
                    "stock_values": int(counter.stock.sum() + counter.stock_nan.sum()),
                    "direct_values": int(counter.direct.sum() + counter.direct_nan.sum()),
                    "different_histogram_bins": int(np.count_nonzero(counter.stock != counter.direct)),
                    "unpaired_chunks": int(counter.unpaired_chunks.sum()),
                    "exact_mismatches": int(counter.exact_mismatches.sum()),
                    "max_abs_residual": float(counter.max_abs_residual.max(initial=0.0)),
                }
        energy = self.energy_distributions.comparison() if self.energy_distributions is not None else None
        if energy is not None:
            summary["sr_energy"] = energy
        categorical_differences = sum(
            a != c for states in self.tallies.values() for codes in states.values() for a, c in codes.values()
        )
        passed = (energy is None or energy["passed"]) and not categorical_differences and all(
            row["stock_values"] == row["direct_values"]
            and not (row["different_histogram_bins"] or row["unpaired_chunks"] or row["exact_mismatches"])
            for row in summary.values()
        )
        return {"passed": passed, "observables": summary, "different_code_tallies": categorical_differences}

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
        if self.energy_distributions is not None:
            collections["sr"]["observables"]["energy"] = self.energy_distributions.persist(arrays)
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
                "energy_method": self.energy_identity,
                "stock_availability": self.stock_availability,
            }, output, indent=2, allow_nan=False)
            output.write("\n")
