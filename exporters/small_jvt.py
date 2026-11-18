"""Export the Run 3 EMPFlow FixedEffPt NNJvt corrections from pinned CVMFS data.

The native provider uses the AnalysisBase FastGraph input order and float
boundaries. The scale-factor histograms become fixed-bin MultiBinning children.
"""
import hashlib
import json
import math
from pathlib import Path


DEFAULT_GROUP_DATA = Path("/cvmfs/atlas.cern.ch/repo/sw/database/GroupData")
NNJVT_AREA = "JetPileupTag/NNJvt/2022-03-22"
SOURCE_FILES = {
    "network": (
        f"{NNJVT_AREA}/NNJVT.Network.graph.Offline.Nonprompt_All_MaxWeight.json",
        "60abc79f5e724483f61967cd5bb2083c2c71255a62e6813f387c1d523dabecdb",
    ),
    "cuts": (
        f"{NNJVT_AREA}/NNJVT.Cuts.FixedEffPt.Offline.Nonprompt_All_MaxW.json",
        "a29154427f1da0c33095a587fcb2d110e63203713d0f39a4bc54a71f680f2e56",
    ),
    "scale_factors": (
        "JetJvtEfficiency/May2024/NNJvtSFFile_Run3_EMPFlow.root",
        "732a52c13ae0dd9a51de3e3f53b10150e1426a4dddb52e99e3884770377cf10b",
    ),
}
ATHENA_REVISION = "6ce3f7ae7c41dd9eaec170c3c349df15faef7e5c"
EVALUATOR_REVISION = "3dcfa20ec01fb40c4d5fec58383785519dec1c00"
WORKING_POINT = "FixedEffPt"
WINDOW = {"min_pt": 20_000.0, "max_pt": 60_000.0, "min_abs_eta": -1.0, "max_abs_eta": 2.5}
OUTPUTS = {
    "small_nnjvt_update": ["score", "rpt", "decorator_pass", "status"],
    "small_nnjvt_select": ["pass", "status"],
    "small_nnjvt_sf": ["sf", "status", "hist_weight"],
}


def verify_sources(group_data):
    receipts = {}
    source_bytes = {}
    for role, (relative, expected) in SOURCE_FILES.items():
        path = Path(group_data) / relative
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise ValueError(f"{path}: expected SHA-256 {expected}, got {digest}")
        receipts[role] = {
            "logical_path": relative, "resolved_path": str(path.resolve()),
            "sha256": digest, "bytes": len(data),
        }
        source_bytes[role] = data
    return receipts, source_bytes


def encode_cut_map(raw):
    source = json.loads(raw)
    pt_edges = source["ptbin_edges"]
    eta_edges = source["etabin_edges"]
    cells = source["cuts"]
    if len(cells) != (len(pt_edges) - 1) * (len(eta_edges) - 1):
        raise ValueError("expected a complete NNJvt cut grid")
    return {
        "pt_edges_gev": pt_edges,
        "eta_edges": eta_edges,
        "cuts": [
            [cells[f"({i}, {j})"] for j in range(len(eta_edges) - 1)]
            for i in range(len(pt_edges) - 1)
        ],
    }


def explicit_edges(axis):
    edges = axis.GetXbins()
    if edges.GetSize() != axis.GetNbins() + 1:
        raise ValueError("NNJvt scale factors require explicit variable ROOT axes")
    result = [float(edges.At(i)) for i in range(edges.GetSize())]
    if any(not math.isfinite(edge) for edge in result) or any(
        high <= low for low, high in zip(result, result[1:])
    ):
        raise ValueError("invalid NNJvt scale-factor edges")
    return result


def histogram_table(histogram, name):
    if not histogram or histogram.ClassName() != "TH2D":
        raise ValueError(f"missing TH2D {name}")
    axes = [explicit_edges(histogram.GetXaxis()), explicit_edges(histogram.GetYaxis())]
    contents, errors = [], []
    # CorrectionLib MultiBinning uses C order: Y is the fastest index.
    for x in range(1, len(axes[0])):
        for y in range(1, len(axes[1])):
            bin_id = histogram.GetBin(x, y)
            contents.append(float(histogram.GetBinContent(bin_id)))
            errors.append(float(histogram.GetBinError(bin_id)))
    if not all(map(math.isfinite, contents + errors)):
        raise ValueError(f"non-finite NNJvt input histogram {name}")
    return axes, contents, errors


def histogram_children(root, path):
    source = root.TFile.Open(str(path), "READ")
    if not source:
        raise OSError(f"cannot open {path}")
    try:
        if source.IsZombie():
            raise OSError(f"cannot read {path}")
        sf_name = f"Jvt{WORKING_POINT}"
        eff_name = f"Eff{WORKING_POINT}"
        edges, sf, sf_error = histogram_table(source.Get(sf_name), sf_name)
        eff_edges, eff, eff_error = histogram_table(source.Get(eff_name), eff_name)
        if edges != eff_edges:
            raise ValueError("SF and efficiency histogram axes disagree")
    finally:
        source.Close()

    values = [[1.0] * len(sf), sf, sf_error, eff, eff_error]
    return [
        {"nodetype": "multibinning", "inputs": ["pt", "eta"], "edges": edges,
         "content": content, "flow": 0.0}
        for content in values
    ]


def variable(name, kind, description):
    return {"name": name, "type": kind, "description": description}


def correction(name, node, inputs, config, description, children=()):
    shape = [len(OUTPUTS[name])]
    return {
        "name": name, "version": 1, "description": description, "inputs": inputs,
        "output": {
            "name": "result", "type": "real", "shape": shape,
            "description": ", ".join(OUTPUTS[name]),
        },
        "data": {
            "nodetype": "extension", "version": 1, "type": f"org.atlas.jets.{node}",
            "output_shape": shape, "config": config, "children": list(children),
        },
    }


def export_small_jvt(group_data=DEFAULT_GROUP_DATA, *, root=None):
    """Return correction definitions and their separate provenance and adapter contract."""
    if root is None:
        import ROOT as root

    sources, data = verify_sources(group_data)
    cuts = {"cut_map": encode_cut_map(data["cuts"]), "mev_per_gev": 1000.0}
    pt_eta = [
        variable("pt", "real", "Float-stored pT in MeV (promoted to double)"),
        variable("eta", "real", "Float-stored signed jet eta"),
    ]
    update = correction(
        "small_nnjvt_update", "nnjvt_update",
        [
            *pt_eta,
            variable("sum_pt_trk", "real", "Float SumPtTrkPt500 at the source primary vertex"),
            variable("jvf_corr", "real", "Float JVFCorr"),
            variable("vertex_kind", "int", "0 absent/invalid, 1 single NoVtx, 2 PriVtx"),
            variable("feature_status", "int", "-1 missing/error, 0 empty track vector, 1 available"),
        ],
        {**cuts, "max_pt": WINDOW["max_pt"], "network_json": data["network"].decode("utf-8")},
        "Run 3 EMPFlow classic FastGraph score, RpT, decorator pass, and update status.",
    )
    selection = correction(
        "small_nnjvt_select", "nnjvt_select",
        [
            *pt_eta,
            variable("score", "real", "Float NNJvt decoration after the attempted update"),
            variable("score_present", "int", "1 if the NNJvt decoration is available, 0 otherwise"),
        ],
        {**WINDOW, **cuts},
        "FixedEffPt working-point selection as a boolean and its status.",
    )
    scale_factor_path = Path(sources["scale_factors"]["resolved_path"])
    weights = correction(
        "small_nnjvt_sf", "nnjvt_sf",
        [
            *pt_eta,
            variable("pass", "int", "Working-point selection boolean"),
            variable("is_jvt_hs", "int", "-1 missing truth label, 0 non-HS, 1 HS"),
            variable("variation", "int", "-1 down, 0 nominal, +1 up"),
            variable("preselected", "int", "Whether JvtEfficiencyAlg processes this jet"),
        ],
        WINDOW,
        "Run 3 FixedEffPt object efficiency/inefficiency SF, status, and histogram weight.",
        histogram_children(root, scale_factor_path),
    )
    provenance = {
        "analysisbase_release": "25.2.117",
        "athena_commit": ATHENA_REVISION,
        "correctionlib_commit": EVALUATOR_REVISION,
        "lwtnn_commit": "7056a1524b07d97d5376f72c20030f831c0dffff",
        "working_point": WORKING_POINT,
        "systematic": "JET_NNJvtEfficiency",
        "model_kind": "lwt::generic::FastGraph<double>",
        "model_inputs": ["Rpt", "JVFCorr", "ptbin", "etabin"],
        "sources": sources,
        "source_symbols": {
            "update": "JetPileupTag::JetVertexNNTagger",
            "selection": "CP::NNJvtSelectionTool",
            "scale_factor": "CP::NNJvtEfficiencyTool",
            "event_truth_labels": "JetPileupLabelingTool",
        },
        "histogram_objects": [f"Jvt{WORKING_POINT}", f"Eff{WORKING_POINT}"],
        "histogram_children": ["in_range_bin", "sf_content", "sf_error", "eff_content", "eff_error"],
        "outputs": OUTPUTS,
        "status_codes": {
            "-1": "error: abort before writing the original decoration",
            "0": "skipped: no new NNJvt decoration, or SF -1 without a new validity flag",
            "1": "ok",
            "2": "scale factor out of validity: write SF -1 and a false validity flag",
        },
        "status_notes": {
            "update": "On a skip use inherited score availability/value for selection; ignore placeholder outputs.",
            "selection": "Write the successful pass result through the framework as_char accessor.",
            "scale_factor": "Histogram weight is the raw SF if valid, otherwise 1. A status of error aborts.",
        },
    }
    return [update, selection, weights], provenance
