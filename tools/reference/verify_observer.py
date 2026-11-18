"""Validate stock observer coverage, identities, stage shapes and validity availability."""

from collections import Counter
import csv
import json
from pathlib import Path

import awkward as ak
import numpy as np
import typer

from workflow import SELECTION_ACCEPT, SELECTION_REJECT, branch_states, flatten, open_root

APP = typer.Typer(add_completion=False)
FOUR_VECTOR = ("pt", "eta", "phi", "m")
EVENT_KEYS = ("eventNumber", "runNumber", "lumiBlock", "mcChannelNumber")
COLLECTIONS = {"sr": "AnalysisJets", "lr": "AnalysisLargeRJets"}


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def exact(got: ak.Array, expected: ak.Array) -> bool:
    return bool(ak.all(got == expected, axis=None))


def check_tools(file: Path, manifest: dict) -> set[str]:
    with file.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    require(not any(r["list"] == "missing" for r in rows), "An initialized tool was not found")
    recommended = {
        (r["tool"], r["variation"]) for r in rows if r["list"] == "recommended"
    }
    affecting = {(r["tool"], r["variation"]) for r in rows if r["list"] == "affecting"}
    require(recommended <= affecting, "A recommended variation is absent from affecting")
    points = set()
    for _, state in recommended:
        if state.endswith("__continuous"):
            stem = state.removesuffix("__continuous")
            points.update((stem + "__1up", stem + "__1down"))
        else:
            points.add(state)
    require(
        points == (set(manifest["service"]) - {"NOSYS"}),
        "Tool recommendations differ from the global state inventory",
    )
    return {s for tool, s in recommended if "FFSmearingTool" in tool}


def check_coverage(tree, manifest: dict, jmr: set[str]) -> dict[str, list[str]]:
    names = set(tree.keys())
    full = {"sr": manifest["sr_p4"], "lr": manifest["lr_histogram"]}
    for prefix, states in full.items():
        fields = [f"p4{axis}" for axis in FOUR_VECTOR] + ["index"]
        if prefix == "sr":
            fields += ["jesJerValidity", "jvtValidity", "jvtSelection", "nnjvt", "nnjvtPass", "isJvtHS"]
        else:
            fields += ["jesJERJMSValidity", "jmrValidity", "selectPtEta", "selectMass",
                       "truthLabel", "d2", "tau1", "tau2", "tau3", "split12", "split23"]
        for field in fields:
            require(branch_states(names, f"{prefix}_{field}_") == set(states),
                    f"Missing or unexpected {prefix} {field} state")
        uncert = set(states) - jmr
        for axis in FOUR_VECTOR:
            require(branch_states(names, f"{prefix}_uncert{axis}_") == uncert,
                    f"Missing or unexpected {prefix} intermediate uncertainty {axis} state")
    for field in ("jvtEfficiency", "eventSF"):
        require(branch_states(names, f"sr_{field}_") == set(manifest["sr_histogram"]),
                f"Missing or unexpected {field} weight state")
    return full


def count_bits(values: ak.Array) -> Counter:
    array = flatten(values)
    bits, counts = np.unique(array, return_counts=True)
    return Counter({int(bit): int(count) for bit, count in zip(bits, counts, strict=True)})


@APP.command()
def verify(
    observer: Path, original: Path, states_file: Path, initialized_tools: Path,
    output: Path, events: int,
) -> None:
    """Check one observer run against the original PHYSLITE and initialized tools."""
    manifest = json.loads(states_file.read_text())
    jmr = check_tools(initialized_tools, manifest)
    counts = {prefix: Counter() for prefix in COLLECTIONS}
    statuses = {field: Counter() for field in
                ("sr_jesJerValidity", "sr_jvtValidity", "lr_jesJERJMSValidity", "lr_jmrValidity")}
    changes = {prefix: Counter() for prefix in COLLECTIONS}
    with open_root(observer) as reference_file, open_root(original) as original_file:
        tree = reference_file["analysis"]
        source = original_file["CollectionTree"]
        require(tree.num_entries == events, f"Expected {events} events, found {tree.num_entries}")
        full = check_coverage(tree, manifest, jmr)
        original_branches = [f"EventInfoAuxDyn.{key}" for key in EVENT_KEYS]
        for collection in COLLECTIONS.values():
            for coordinate in FOUR_VECTOR:
                original_branches.extend(f"{collection}AuxDyn.{aux}" for aux in
                                         (coordinate, f"JetConstitScaleMomentum_{coordinate}"))
        offset = 0
        for chunk in tree.iterate(step_size=512, how=dict):
            num_events = len(chunk["eventNumber"])
            source_chunk = source.arrays(
                original_branches, entry_start=offset, entry_stop=offset + num_events, how=dict
            )
            for key in EVENT_KEYS:
                require(exact(chunk[key], source_chunk[f"EventInfoAuxDyn.{key}"]),
                        f"Event identity {key} differs starting at entry {offset}")
            for prefix, collection in COLLECTIONS.items():
                stored_pt = source_chunk[f"{collection}AuxDyn.pt"]
                length = ak.num(stored_pt)
                counts[prefix].update(objects=int(ak.sum(length)), empty_events=int(ak.sum(length == 0)))
                indices = ak.local_index(stored_pt)
                for state in full[prefix]:
                    require(exact(chunk[f"{prefix}_index_{state}"], indices),
                            f"Object order {prefix} {state} differs starting at entry {offset}")
                for coordinate in FOUR_VECTOR:
                    stored = source_chunk[f"{collection}AuxDyn.{coordinate}"]
                    constit = source_chunk[f"{collection}AuxDyn.JetConstitScaleMomentum_{coordinate}"]
                    require(exact(chunk[f"{prefix}_stored{coordinate}_NOSYS"], stored),
                            f"Stored {prefix} {coordinate} differs at entry {offset}")
                    require(exact(chunk[f"{prefix}_const{coordinate}_NOSYS"], constit),
                            f"Constituent {prefix} {coordinate} differs at entry {offset}")
                    calib = chunk[f"{prefix}_calib{coordinate}_NOSYS"]
                    changes[prefix][f"changed_{coordinate}_from_stored"] += int(ak.sum(stored != calib))
                for branch, values in chunk.items():
                    if branch.startswith(prefix + "_") and not branch.startswith("sr_eventSF_"):
                        require(exact(ak.num(values), length),
                                f"Shape mismatch for {branch} starting at entry {offset}")
            for field, status_counts in statuses.items():
                for state in full[field[:2]]:
                    observed = count_bits(chunk[f"{field}_{state}"])
                    if field == "lr_jmrValidity" and state not in jmr:
                        allowed = {0}
                    else:
                        allowed = {int(SELECTION_ACCEPT), int(SELECTION_REJECT)}
                    require(set(observed) <= allowed, f"Unexpected {field} availability in {state}")
                    status_counts.update(observed)
            offset += num_events

        result = {
            "events": offset,
            "source_file_uuid": str(original_file.file.uuid),
            "observer_branches": len(tree.keys()),
            "collections": {p: dict(c) for p, c in counts.items()},
            "changed_from_stored": {p: dict(c) for p, c in changes.items()},
            "status_values": {p: dict(c) for p, c in statuses.items()},
            "jmr_states_decorated": sorted(jmr),
            "status_encoding": {"0": "absent", str(SELECTION_ACCEPT): "valid",
                                str(SELECTION_REJECT): "out_of_validity"},
        }

    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    APP()
