"""Read only the original PHYSLITE branches required by the selected stages."""

from dataclasses import dataclass

import awkward as ak
import numpy as np
from uproot import open as uproot_open
from uproot.source.file import MemmapSource
from uproot.source.futures import TrivialExecutor

P4_FIELDS = ("pt", "eta", "phi", "m")
EVENT_KEYS = ("eventNumber", "runNumber", "lumiBlock", "mcChannelNumber")
COLLECTIONS = {"sr": "AnalysisJets", "lr": "AnalysisLargeRJets"}
# RhoKey in the active area-00-04-83 small-R nominal recommendation.
SMALL_RHO = "Kt4EMPFlowNeutEventShape"
R4_TRUTH = "AntiKt4TruthDressedWZJets"
R10_TRUTH = "AntiKt10TruthSoftDropBeta100Zcut10Jets"
LARGE_LABEL = "R10TruthLabel_R21Precision_2022v1"
LARGE_OBSERVABLES = {
    "truthLabel": LARGE_LABEL,
    "d2": "D2",
    "tau1": "Tau1_wta",
    "tau2": "Tau2_wta",
    "tau3": "Tau3_wta",
    "split12": "Split12",
    "split23": "Split23",
}


def open_root(path):
    return uproot_open(
        path,
        handler=MemmapSource,
        decompression_executor=TrivialExecutor(),
        interpretation_executor=TrivialExecutor(),
        num_fallback_workers=1,
    )


def aux_paths(collection, field):
    return (f"{collection}AuxDyn.{field}", f"{collection}Aux.{field}")


def flatten(values, dtype=None):
    return np.asarray(ak.to_numpy(ak.flatten(values, axis=1)), dtype=dtype)


def offsets(counts):
    terminal = np.empty(len(counts) + 1, dtype=np.int64)
    terminal[0] = 0
    np.cumsum(counts, out=terminal[1:])
    return terminal


def event_rows(counts):
    return np.repeat(np.arange(len(counts), dtype=np.int64), counts)


def read_p4(source, collection, prefix=""):
    return np.column_stack([flatten(source[collection, prefix + coordinate], np.float64)
                            for coordinate in P4_FIELDS])


@dataclass(frozen=True)
class Branch:
    options: tuple[str, ...]
    optional: bool = False


def branch_plan(collections, mode, diagnostics=False, stage="final"):
    plan = {}

    def add(collection, *fields, optional=False):
        for field in fields:
            plan[collection, field] = Branch(aux_paths(collection, field), optional)

    diagnostics = diagnostics or stage != "final"
    if diagnostics:
        add("EventInfo", *EVENT_KEYS)

    for prefix in collections:
        prefix_collection = COLLECTIONS[prefix]
        add(prefix_collection, *(P4_FIELDS if mode != "stored" or diagnostics else ("pt",)))
        if mode != "stored" and stage != "stored":
            add(prefix_collection, *(f"JetConstitScaleMomentum_{coordinate}" for coordinate in P4_FIELDS))
            if stage != "const":
                add(prefix_collection, "DetectorEta")

    if mode == "stored" or stage in ("stored", "const"):
        return plan

    needs_application = stage in ("uncert", "final")
    if "sr" in collections or needs_application:
        add("EventInfo", "averageInteractionsPerCrossing")
        add("EventInfo", "PVIndex", "NPV", optional=True)
        add("PrimaryVertices", "vertexType")
        plan["PrimaryVertices", "trackParticleLinks"] = Branch(
            (*aux_paths("PrimaryVertices", "trackParticleLinks"),
             *aux_paths("PrimaryVertices", "trackParticleLinks.m_persIndex")),
            optional=True,
        )
    if "sr" in collections:
        small = COLLECTIONS["sr"]
        add(SMALL_RHO, "Density")
        add(small, "ActiveArea4vec_pt")
        if stage != "pileup" and stage != "etaJES":
            add(small, "SumPtChargedPFOPt500", "NumTrkPt1000", "TrackWidthPt1000", "EnergyPerSampling")
        if needs_application:
            add(small, "PartonTruthLabelID", optional=True)
        if stage == "final":
            add(small, "SumPtTrkPt500", "JVFCorr")
            add(small, "NNJvt", "NNJvtPass", optional=True)

    if "lr" in collections and needs_application:
        large = COLLECTIONS["lr"]
        add(large, "PartonTruthLabelID", optional=True)
        add(large, LARGE_LABEL)
        add(large, *(f"JetJMSScaleMomentumCalo_{coordinate}" for coordinate in P4_FIELDS), optional=True)
        if diagnostics and stage == "final":
            add(large, *[field for field in LARGE_OBSERVABLES.values() if field != LARGE_LABEL])
        if stage == "final" and mode == "full":
            add(R10_TRUTH, *P4_FIELDS)

    if stage == "final":
        # The stock pre-jet pile-up labeller runs on both jet collections.
        add(R4_TRUTH, *P4_FIELDS)
    return plan


def resolve_branches(tree, plan):
    available = set(tree.keys())
    resolved = {}
    for label, branch in plan.items():
        physical = next((name for name in branch.options if name in available), None)
        if physical is None and not branch.optional:
            raise KeyError(f"Missing PHYSLITE branch for {label}: {branch.options}")
        resolved[label] = physical
    return resolved


def read_chunk(tree, resolved, start, stop):
    selected = sorted({branch for branch in resolved.values() if branch is not None})
    arrays = tree.arrays(selected, entry_start=start, entry_stop=stop, how=dict, library="ak")
    return {label: arrays[branch] for label, branch in resolved.items() if branch is not None}


def optional_event(inputs, field, events, dtype, default):
    values = inputs.get(("EventInfo", field))
    if values is None:
        return np.full(events, default, dtype=dtype), 0
    return np.asarray(ak.to_numpy(values), dtype=dtype), 1


def optional_object(inputs, collection, field, count, dtype, default):
    values = inputs.get((collection, field))
    if values is None:
        return np.full(count, default, dtype=dtype), 0
    return flatten(values, dtype), 1


def track_counts(inputs):
    links = inputs.get(("PrimaryVertices", "trackParticleLinks"))
    if links is None:
        if ("EventInfo", "NPV") not in inputs:
            raise KeyError("PrimaryVertices trackParticleLinks is required to derive uncertainty NPV")
        return np.zeros(len(flatten(inputs["PrimaryVertices", "vertexType"])), dtype=np.int64)
    fields = ak.fields(links)
    if fields:
        links = links[fields[0]]
    return flatten(ak.num(links, axis=2), np.int64)


def pick_feature(inputs, collection, field, indexes, dtype):
    """Gather one inner vector element per original jet with its availability."""
    vectors = ak.flatten(inputs[collection, field], axis=1)
    local = ak.local_index(vectors, axis=1)
    selected = ak.firsts(vectors[local == ak.Array(indexes)[:, np.newaxis]], axis=1)
    available = np.asarray(ak.to_numpy(~ak.is_none(selected)), dtype=bool)
    return np.asarray(ak.to_numpy(ak.fill_none(selected, 0)), dtype=dtype), available


def sampling_features(inputs, collection, fields):
    vectors = ak.flatten(inputs[collection, "EnergyPerSampling"], axis=1)
    # Pad only for shape-safe indexing. A missing active sampling input is an
    # invalid file/recipe match for the pinned PHYSLITE collection.
    if not bool(ak.all(ak.num(vectors, axis=1) > max(fields))):
        raise ValueError(f"EnergyPerSampling for {collection} omits a configured sampling")
    return [np.asarray(ak.to_numpy(vectors[:, index]), dtype=np.float64) for index in fields]
