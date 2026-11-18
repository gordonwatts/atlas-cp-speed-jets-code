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
                result[leaf("eta")] = (leaf("p4eta"),)
                result[leaf("phi")] = (leaf("p4phi"),)
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


def main(
    proof: Path,
    output: Path,
    energy_library: Annotated[Path, typer.Option()],
    chunk: Annotated[int, typer.Option(min=1)] = 512,
):
    """Check aliases and float energy using the exact C comparison as the first leg."""
    execution = check_execution()
    if output.exists():
        raise FileExistsError(output)
    raw_proof = proof.read_bytes()
    report = json.loads(raw_proof)
    if not report["passed"] or report["entries_compared"] != report["entries_requested"]:
        raise ValueError("A passed and complete C observer comparison is required")
    mapping = aliases(report["states"])
    if any(ref not in report["checks"] for refs in mapping.values() for ref in refs):
        raise ValueError("The base comparison omitted a required source field")
    if set(mapping) & set(report["checks"]):
        raise ValueError("A mapped alias overlaps a directly verified field")
    required = set(EVENT_KEYS) | mapping.keys() | {ref for refs in mapping.values() for ref in refs}
    diagnostic = Diagnostics({key: "energy" if len(refs) == 4 else "alias" for key, refs in mapping.items()})
    energy = Energy(energy_library)
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
                    else:
                        values = flatten(first)
                diagnostic.compare(name, data[name], values, identities, start, counts)
            check_execution()
    failures = sum(bool(item["exact_mismatches"] or item["shape_mismatch_events"])
                   for item in diagnostic.checks.values())
    with output.open("x", encoding="utf-8") as stream:
        json.dump({
            "passed": failures == 0, "entries": entries, "execution": execution,
            "authority": report["authority"], "base_comparison": str(proof.resolve()),
            "base_comparison_sha256": sha256(raw_proof).hexdigest(),
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
    if failures:
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
