"""Prove the extra stock output names from fields already checked against C."""

from ctypes import CDLL, c_size_t
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Annotated

CHECKOUT = Path(__file__).resolve().parents[1]
if str(CHECKOUT) not in sys.path:
    sys.path.insert(0, str(CHECKOUT))

from columnar.inputs import EVENT_KEYS, flatten, open_root
from columnar.jets import WEIGHT_DIRECTIONS, check_execution
from tools.compare_columnar import Diagnostics

import awkward as ak
import numpy as np
import typer

RELEASE = "AnalysisBase 25.2.117 (Athena 6ce3f7ae7c41dd9eaec170c3c349df15faef7e5c)"
SOURCE = [
    "PhysicsAnalysis/Algorithms/AsgAnalysisAlgorithms/Root/AsgEnergyDecoratorAlg.cxx:27-28",
    "Event/xAOD/xAODJet/Root/Jet_v1.cxx:63-65",
    "PhysicsAnalysis/Algorithms/JetAnalysisAlgorithms/python/JetAnalysisConfig.py:559-563",
    "PhysicsAnalysis/Algorithms/JetAnalysisAlgorithms/python/JetJvtAnalysisConfig.py:51",
    "config/jets-observer.yaml",
]


def aliases(states):
    result = {}
    for prefix, names in states.items():
        for state in names:
            def leaf(field):
                return f"{prefix}_{field}_{state}"

            if prefix == "sr":
                result[f"weight_jvt_effSF_{state}"] = (leaf("eventSF"),)
                if state in WEIGHT_DIRECTIONS:
                    continue
                result[leaf("select_baselineJvt")] = (leaf("jvtSelection"),)
                result[leaf("e")] = tuple(leaf(field) for field in ("p4pt", "p4eta", "p4phi", "p4m"))
            else:
                result[leaf("m")] = (leaf("p4m"),)
            result[leaf("pt")] = (leaf("p4pt"),)
            if state == "NOSYS":
                result[f"{prefix}_eta"] = (leaf("p4eta"),)
                result[f"{prefix}_phi"] = (leaf("p4phi"),)
    return result


class Energy:
    def __init__(self, path):
        self.library = CDLL(str(path.resolve(strict=True)))
        self.evaluate = self.library.columnar_alias_energy
        self.evaluate.restype = None
        f32 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
        self.evaluate.argtypes = [c_size_t, f32, f32, f32, f32, f32]

    def __call__(self, p4):
        coordinates = [np.ascontiguousarray(flatten(values), dtype=np.float32) for values in p4]
        output = np.empty_like(coordinates[0])
        self.evaluate(len(output), *coordinates, output)
        return output


class EnergyDistributions:
    def __init__(self, states):
        from columnar.distributions import Distribution, Observable

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

    def write(self, output, entries, sources, absence, proof):
        from columnar.distributions import RESIDUAL

        output.mkdir()
        counter = self.distribution
        arrays = {"entries": np.asarray(entries), "sr_states": np.asarray(self.states),
                  "paired_residual_edges": np.asarray(RESIDUAL)}
        metadata = counter.persist("sr_energy", arrays, self.states)
        arrays["sr_energy_stock_finite_range"] = self.ranges[:, 0]
        arrays["sr_energy_direct_finite_range"] = self.ranges[:, 1]
        samples_per_side_equal = np.array_equal(counter.stock.sum(axis=1) + counter.stock_nan,
                                                counter.direct.sum(axis=1) + counter.direct_nan)
        passed = (
            samples_per_side_equal
            and np.array_equal(counter.stock, counter.direct)
            and np.array_equal(counter.stock_nan, counter.direct_nan)
            and np.array_equal(self.ranges[:, 0], self.ranges[:, 1])
            and not (np.any(counter.unpaired_chunks) or np.any(counter.exact_mismatches))
        )
        with (output / "distributions.npz").open("xb") as stream:
            np.savez(stream, **arrays)
        with (output / "metadata.json").open("x", encoding="utf-8") as stream:
            json.dump({
                "passed": passed,
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
        return {"path": str(output.resolve()), "passed": passed, "states": len(self.states),
                "values_per_side": int(counter.stock.sum() + counter.stock_nan.sum())}


def observer_availability(keys):
    return {
        "large_r_energy": {"persisted": any(name == "lr_e" or name.startswith("lr_e_") for name in keys),
                           "alternative": "per-object pt, eta, phi and mass are persisted and verified"},
        "large_r_unmodified_auxiliary_branches": {
            field: sum(name.startswith(f"lr_{field}_") for name in keys)
            for field in ("d2", "tau1", "tau2", "tau3", "split12", "split23", "truthLabel")
        },
    }


def main(
    proof: Path,
    output: Path,
    energy_library: Annotated[Path, typer.Option()],
    chunk: Annotated[int, typer.Option(min=1)] = 512,
    also_proof: Annotated[list[Path] | None, typer.Option(help="Proven comparison of the same stock fields")] = None,
    energy_distributions: Annotated[Path | None, typer.Option(help="Write independently counted energy spectra")] = None,
):
    """Check aliases and float energy using the exact C comparison as the first leg."""
    execution = check_execution()
    if output.exists():
        raise FileExistsError(output)
    if energy_distributions is not None and energy_distributions.exists():
        raise FileExistsError(energy_distributions)
    proofs = {}
    report = None
    for path in [proof, *(also_proof or [])]:
        raw_proof = path.read_bytes()
        checked = json.loads(raw_proof)
        if not checked["passed"] or checked["entries_compared"] != checked["entries_requested"]:
            raise ValueError(f"A passed and complete C observer comparison is required: {path}")
        if report is None:
            report = checked
        elif (any(checked[key] != report[key] for key in ("authority", "source_file_uuid", "entries_compared", "states"))
              or checked["checks"] != report["checks"]):
            raise ValueError(f"The proof must match the stock file, interval, and directly checked fields: {path}")
        proofs[str(path.resolve())] = sha256(raw_proof).hexdigest()
    mapping = aliases(report["states"])
    if any(ref not in report["checks"] for refs in mapping.values() for ref in refs):
        raise ValueError("The base comparison omitted a required source field")
    if set(mapping) & set(report["checks"]):
        raise ValueError("A mapped alias overlaps a directly verified field")
    required = set(EVENT_KEYS) | mapping.keys() | {ref for refs in mapping.values() for ref in refs}
    diagnostic = Diagnostics({key: "energy" if len(refs) == 4 else "alias" for key, refs in mapping.items()})
    energy = Energy(energy_library)
    energy_counts = None
    if energy_distributions is not None:
        energy_counts = EnergyDistributions(
            [state for state in report["states"].get("sr", []) if f"sr_e_{state}" in mapping])
    entries = report["entries_compared"]
    with open_root(report["authority"]) as source:
        tree = source["analysis"]
        persisted = set(tree.keys())
        missing = required - persisted
        if missing or entries > tree.num_entries:
            raise ValueError(f"Stock observer does not cover alias proof: {sorted(missing)[:8]}")
        covered = set(report["checks"]) | set(mapping)
        if report["mode"] == "full" and report["scope"] == "both" and covered != persisted:
            uncovered = persisted - covered
            unknown = covered - persisted
            raise ValueError(f"Complete observer coverage failed: {sorted(uncovered)[:8]=}, {sorted(unknown)[:8]=}")
        availability = observer_availability(persisted)
        for start in range(0, entries, chunk):
            stop = min(start + chunk, entries)
            data = tree.arrays(sorted(required), entry_start=start, entry_stop=stop, how=dict, library="ak")
            identities = {name: np.asarray(ak.to_numpy(data[name])) for name in EVENT_KEYS}
            for name, refs in mapping.items():
                source_fields = [data[ref] for ref in refs]
                first = source_fields[0]
                if name.startswith("weight_jvt_effSF_"):
                    counts, values = None, np.asarray(ak.to_numpy(first))
                else:
                    counts = np.asarray(ak.to_numpy(ak.num(first, axis=1)), dtype=np.int64)
                    if len(refs) == 4:
                        for other in source_fields[1:]:
                            np.testing.assert_array_equal(counts, ak.to_numpy(ak.num(other, axis=1)))
                        values = energy(source_fields)
                        if energy_counts is not None:
                            stored_counts = ak.to_numpy(ak.num(data[name], axis=1))
                            energy_counts.fill(name, flatten(data[name]), values, np.array_equal(stored_counts, counts))
                    else:
                        values = flatten(first)
                diagnostic.compare(name, data[name], values, identities, start, counts)
            check_execution()
    failures = sum(bool(item["exact_mismatches"] or item["shape_mismatch_events"])
                   for item in diagnostic.checks.values())
    spectrum = (energy_counts.write(energy_distributions, entries, proofs, availability, output)
                if energy_counts is not None else None)
    passed = failures == 0 and (spectrum is None or spectrum["passed"])
    with output.open("x", encoding="utf-8") as stream:
        json.dump({
            "passed": passed, "entries": entries, "execution": execution,
            "energy_distributions": spectrum, "stock_availability": availability,
            "authority": report["authority"], "base_comparisons": proofs,
            "observer_branches": len(diagnostic.checks),
            "complete_branches": len(report["checks"]) + len(diagnostic.checks),
            "comparisons": sum(row["comparisons"] for row in diagnostic.checks.values()),
            "energy_library": str(energy_library.resolve()),
            "energy_library_sha256": sha256(energy_library.read_bytes()).hexdigest(),
            "source": {"release": RELEASE, "files": SOURCE},
            "mapping": {key: list(refs) for key, refs in mapping.items()},
            "checks": diagnostic.checks,
        }, stream, indent=2, allow_nan=False)
        stream.write("\n")
    typer.echo(f"{entries} events, {len(mapping)} supplementary stock fields, {failures} mismatches")
    if not passed:
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
