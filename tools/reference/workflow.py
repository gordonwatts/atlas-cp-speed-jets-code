"""Read the stock AnalysisBase jet ntuple and write the pT-by-systematic histograms."""

import json
from pathlib import Path

import awkward as ak
import hist
import numpy as np
import typer
import uproot
from uproot.source.file import MemmapSource
from uproot.source.futures import TrivialExecutor

APP = typer.Typer(no_args_is_help=True)
SELECTION_ACCEPT = np.uint32(0xFFFFFFFF)
SELECTION_REJECT = np.uint32(0xFFFFFFFE)


def open_root(path: Path):
    return uproot.open(
        path,
        handler=MemmapSource,
        num_fallback_workers=1,
        decompression_executor=TrivialExecutor(),
        interpretation_executor=TrivialExecutor(),
    )


def branch_states(branches: set[str], prefix: str) -> set[str]:
    return {name.removeprefix(prefix) for name in branches if name.startswith(prefix)}


@APP.command()
def inventory(ntuple: Path, output: Path) -> None:
    """Freeze the release's global ordering and actually written observable states."""
    with open_root(ntuple) as root_file:
        tree = root_file["analysis"]
        branches = set(tree.keys())
        service = root_file["systematicsList"].axis().labels()
        weight = root_file["systematicsListOnlyWeights"].axis().labels()
        kin = [state for state in service if state not in weight]
        small_pt = branch_states(branches, "sr_pt_")
        small_sf = branch_states(branches, "sr_jvtEfficiency_")
        large_pt = branch_states(branches, "lr_pt_")
        assert (small_pt | large_pt) == set(kin)
        assert (small_sf - small_pt) == set(weight)
        assert large_pt == branch_states(branches, "lr_selectMass_")
        assert large_pt == branch_states(branches, "lr_selectPtEta_")
        manifest = {
            "service": service,
            "service_kinematic": kin,
            "service_weight_only": weight,
            "sr_p4": [state for state in kin if state in small_pt],
            "sr_histogram": [state for state in kin + weight if state in small_sf],
            "lr_histogram": [state for state in kin if state in large_pt],
            "correlated_kinematic": [state for state in kin if state in small_pt & large_pt],
            "entries_observed": tree.num_entries,
        }
    with output.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")


def choose_names(mode: str, manifest: dict) -> tuple[list[str], list[str]]:
    if mode == "full":
        return manifest["sr_histogram"], manifest["lr_histogram"]
    if mode in ("nominal", "stored"):
        return ["NOSYS"], ["NOSYS"]
    raise ValueError(f"Unknown mode {mode!r}")


def new_histogram(edges: np.ndarray, states: list[str]):
    return hist.Hist(
        hist.axis.Variable(edges, name="pt"),
        hist.axis.StrCategory(states, name="state", overflow=False),
        storage=hist.storage.Weight(),
    )


def flatten(array: ak.Array) -> np.ndarray:
    return ak.to_numpy(ak.flatten(array))


@APP.command("histogram")
def write_histogram(
    ntuple: Path, inventory_file: Path, specification: Path, output: Path, mode: str
) -> None:
    """Reread stock pT, scale factor and selection vectors, then save both histograms."""
    manifest = json.loads(inventory_file.read_text())
    spec = json.loads(specification.read_text())
    edges = np.asarray(spec["bins_gev"], dtype=np.float64)
    small_states, large_states = choose_names(mode, manifest)
    small_kin = set(manifest["sr_p4"])
    small = new_histogram(edges, small_states)
    large = new_histogram(edges, large_states)
    small_entries = np.zeros(len(small_states), dtype=np.int64)
    large_entries = np.zeros(len(large_states), dtype=np.int64)

    with open_root(ntuple) as root_file:
        tree = root_file["analysis"]
        branches = {f"lr_pt_{state}" for state in large_states}
        branches |= {f"sr_pt_{state}" for state in small_states if state in small_kin}
        if mode != "stored":
            branches |= {f"sr_jvtEfficiency_{state}" for state in small_states}
            branches |= {f"sr_jvtValidity_{state}" for state in small_states if state in small_kin}
            branches |= {f"lr_{cut}_{state}" for cut in ("selectPtEta", "selectMass")
                         for state in large_states}
        for chunk in tree.iterate(sorted(branches), step_size=spec["chunk_events"], how=dict):
            for index, state in enumerate(small_states):
                kinematic = state if state in small_kin else "NOSYS"
                pt = flatten(chunk[f"sr_pt_{kinematic}"])
                weights = None
                if mode != "stored":
                    raw = flatten(chunk[f"sr_jvtEfficiency_{state}"])
                    validity = flatten(chunk[f"sr_jvtValidity_{kinematic}"])
                    if len(raw) != len(pt) or len(validity) != len(pt):
                        raise ValueError(f"Small-R shape mismatch for {state}")
                    if not np.isin(validity, (SELECTION_ACCEPT, SELECTION_REJECT)).all():
                        raise ValueError(f"Unknown NNJVT validity for {state}")
                    if not np.all(raw[validity == SELECTION_REJECT] == -1):
                        raise ValueError(f"Invalid NNJVT scale factor sentinel for {state}")
                    weights = np.where(validity == SELECTION_ACCEPT, raw, 1.0)
                small.fill(pt=pt.astype(np.float64) / 1000, state=state, weight=weights)
                small_entries[index] += len(pt)
            for index, state in enumerate(large_states):
                pt = flatten(chunk[f"lr_pt_{state}"])
                if mode != "stored":
                    pt_eta = chunk[f"lr_selectPtEta_{state}"] == SELECTION_ACCEPT
                    mass = chunk[f"lr_selectMass_{state}"] == SELECTION_ACCEPT
                    pt = flatten(chunk[f"lr_pt_{state}"][pt_eta & mass])
                large.fill(pt=pt.astype(np.float64) / 1000, state=state)
                large_entries[index] += len(pt)

        events = tree.num_entries

    with output.open("xb") as stream:
        np.savez(
            stream, edges=edges, events=events,
            sr_states=np.asarray(small_states), sr_entries=small_entries,
            sr_sumw=small.values(flow=True), sr_sumw2=small.variances(flow=True),
            lr_states=np.asarray(large_states), lr_entries=large_entries,
            lr_sumw=large.values(flow=True), lr_sumw2=large.variances(flow=True),
        )


if __name__ == "__main__":
    APP()
