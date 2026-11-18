"""Export both initialized MC23 Category/Full jet uncertainty recommendations."""

import csv
import json
import re
from pathlib import Path

from exporters.calibdata import read_config, source_record
from exporters.rootdata import export_histogram

ATHENA_COMMIT = "6ce3f7ae7c41dd9eaec170c3c349df15faef7e5c"
CALIB_AREA = "CalibArea-08"
MC_TYPE = "MC23"
GROUP_PREFIX = "JET_"
MODELS = {
    "SmallR": {
        "jet": "AntiKt4EMPFlow",
        "config": "rel22/Winter2025_PreRec/R4_CategoryReduction_FullJER.config",
        "correction": "small_uncertainty",
        "contract_key": "sr_p4",
    },
    "LargeR": {
        "jet": "AntiKt10UFOCSSKSoftDropBeta100Zcut10",
        "config": "rel22/Summer2025_PreRec/R10_CategoryJES_FullJER_FullJMS.config",
        "correction": "large_uncertainty",
        "contract_key": "lr_histogram",
    },
}
SOURCE_NUMERICS = {
    "mev_to_gev": 1e-3,
    "rng_user_seed": 0,
    "mc_phi_seed_multiplier": 2e5,
    "pseudo_data_phi_seed_multiplier": 1.23e5,
    "mc_zero_seed": 45583453,
    "pseudo_data_zero_seed": 34545654,
}


def resolve_uncertainty_source(resolver, source_name):
    if Path(source_name).is_file():
        return source_name, Path(source_name)
    for requested in (f"JetUncertainties/{CALIB_AREA}/{source_name}", source_name):
        resolved = str(resolver.FindCalibFile(requested))
        if resolved and Path(resolved).is_file():
            return requested, Path(resolved)
    raise FileNotFoundError(f"could not resolve uncertainty source {source_name!r}")


def tables_needed(settings, jet, mc_type):
    """Source-name derivation in ConfigHelper and ResolutionHelper."""
    names = set()

    def collect(raw_names):
        for name in re.split(r"[ ,]+", raw_names):
            if name:
                names.add(f"{name.replace('MCTYPE', mc_type)}_{jet}")

    for key, name in settings.items():
        match = re.fullmatch(r"(JESComponent\.\d+\.)Name", key)
        if not match or not name:
            continue
        prefix = match[1]
        histograms = settings.get(prefix + "Hists") or settings.get(prefix + "SubComp") or name
        collect(histograms)
        collect(settings.get(prefix + "VHist", ""))

    for name in ("FileValidHistogram", "CombMassWeightCaloHist", "CombMassWeightTAHist"):
        collect(settings.get(name, ""))
    for name in ("Pileup.NPVRef", "Pileup.MuRef"):
        value = settings.get(name)
        if value:
            try:
                float(value)
            except ValueError as error:
                raise ValueError(f"the configured campaign supports numeric pileup reference {name}") from error
    for key, value in settings.items():
        if key.startswith("Nominal") and value:
            collect(value.split(",", 1)[0].strip())
    return sorted(names)


def root_payload(root, path, keys):
    root_file = root.TFile.Open(str(path), "READ")
    if not root_file or root_file.IsZombie():
        raise ValueError(f"cannot open release histogram source {path}")
    payload, object_info = {}, {}
    try:
        for name in keys:
            histogram = root_file.Get(name)
            key = root_file.GetKey(name)
            if not histogram or not key or not histogram.InheritsFrom("TH1"):
                raise KeyError(f"missing uncertainty ROOT histogram {name} in {path}")
            payload[name] = export_histogram(histogram)
            object_info[name] = {"class": histogram.ClassName(), "cycle": int(key.GetCycle())}
    finally:
        root_file.Close()
    return payload, object_info


def flavour_keys(root, path, jet):
    root_file = root.TFile.Open(str(path), "READ")
    if not root_file or root_file.IsZombie():
        raise ValueError(f"cannot open release flavour source {path}")
    try:
        names = [key.GetName() for key in root_file.GetListOfKeys()]
    finally:
        root_file.Close()
    fractions, errors = [], []
    for name in names:
        if jet not in name or "valid" in name:
            continue
        if "gluonFractionError" in name:
            errors.append(name)
        elif "gluonFraction" in name:
            fractions.append(name)
    if len(fractions) != 1 or len(errors) != 1 or any("njet" in n.rsplit("_", 1)[-1].lower()
                                                       for n in fractions + errors):
        raise ValueError(f"{path} does not contain one inclusive flavour pair for {jet}")
    return fractions[0], errors[0]


def read_tool_states(path, collection):
    tool = f"seq1_JetUncertaintiesAlg_{collection}.uncertaintiesTool"
    states = {(mode, kind): set() for mode in ("main", "pseudo_data")
              for kind in ("affecting", "recommended")}
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            if row["tool"] == tool:
                mode = "main"
            elif row["tool"] == tool + "PD":
                mode = "pseudo_data"
            else:
                continue
            if not row["variation"].endswith("__continuous"):
                raise ValueError(f"unexpected uncertainty tool variation {row['variation']!r}")
            states[mode, row["list"]].add(row["variation"].removesuffix("__continuous"))
    if not all(states.values()) or any(
            not states[mode, "recommended"] <= states[mode, "affecting"]
            for mode in ("main", "pseudo_data")):
        raise ValueError(f"incomplete initialized uncertainty declarations for {collection}")
    return states


def compile_states(contract, tool_states, model):
    active = {}
    for state in contract["service_kinematic"]:
        if state == "NOSYS":
            continue
        match = re.fullmatch(r"(.+)__1(up|down)", state)
        if not match:
            raise ValueError(f"unsupported frozen systematic name: {state}")
        basename, direction = match.groups()
        mode = "pseudo_data" if "PseudoData" in basename else "main"
        if basename in tool_states[mode, "affecting"]:
            active[state] = {
                "basename": basename.removesuffix("_PseudoData"),
                "parameter": 1 if direction == "up" else -1,
                "pseudo_data": mode == "pseudo_data",
            }
    expected = {state for state in contract[model["contract_key"]]
                if state != "NOSYS" and not state.startswith("JET_JMRUnc_")}
    if active.keys() != expected:
        raise ValueError(f"configured and initialized uncertainty states differ for {model['jet']}: "
                         f"{sorted(active.keys() ^ expected)}")
    return active


def correction_def(name, configuration):
    variables = (
        ("pt", "real"), ("eta", "real"), ("phi", "real"), ("m", "real"),
        ("average_mu", "real"), ("uncertainty_npv", "real"), ("event_available", "int"),
        ("preselected", "int"), ("parton_truth_label", "int"), ("parton_truth_label_available", "int"),
        ("large_r_truth_label", "int"), ("large_r_truth_label_available", "int"),
        ("calo_pt", "real"), ("calo_eta", "real"), ("calo_phi", "real"), ("calo_m", "real"),
        ("calo_available", "int"), ("systematic", "string"),
    )
    return {
        "name": name, "version": 1,
        "description": "AnalysisBase 25.2.117 MC23 Category/Full calibrated jet uncertainties",
        "inputs": [{"name": field, "type": kind} for field, kind in variables],
        "output": {"name": "jet_and_status", "type": "real", "shape": [8],
                   "description": "[pt, eta, phi, m, status, status_available, is_bjet, is_bjet_available]"},
        "data": {"nodetype": "extension", "type": "org.atlas.jets.uncertainty",
                 "version": 1, "output_shape": [8], "config": configuration},
    }


def export_jet_uncertainties(
        contract_path=Path("config/jet-states.json"),
        initialized_path=Path("results/reference/observer-smoke-003/initialized-tools.tsv")):
    """Return both schema-v3 correction definitions and their resolved provenance."""
    import ROOT
    from PathResolver import PathResolver

    contract_path, initialized_path = Path(contract_path), Path(initialized_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    corrections, provenance = [], {}
    for collection, model in MODELS.items():
        cfg_name, cfg_path = resolve_uncertainty_source(PathResolver, model["config"])
        environment = ROOT.TEnv()
        read_config(environment, cfg_path, ROOT.kEnvGlobal)
        settings = {record.GetName(): record.GetValue() for record in environment.GetTable()}
        jet = model["jet"]
        main_name, main_path = resolve_uncertainty_source(PathResolver, settings["UncertaintyRootFile"])
        analysis_name, analysis_path = resolve_uncertainty_source(PathResolver, settings["AnalysisRootFile"])
        histograms, main_objects = root_payload(ROOT, main_path, tables_needed(settings, jet, MC_TYPE))
        fraction, error = flavour_keys(ROOT, analysis_path, jet)
        analysis_histograms, analysis_objects = root_payload(ROOT, analysis_path, (fraction, error))
        tool_states = read_tool_states(initialized_path, collection)
        active_states = compile_states(contract, tool_states, model)
        configuration = {
            "jet_definition": jet, "mc_type": MC_TYPE, "group_prefix": GROUP_PREFIX,
            "settings": settings, "histograms": histograms, "analysis_histograms": analysis_histograms,
            "gluon_fraction_key": fraction, "gluon_fraction_error_key": error,
            "abs_eta_gluon_fraction": True,
            "states": active_states, **SOURCE_NUMERICS,
        }
        corrections.append(correction_def(model["correction"], configuration))
        provenance[collection] = {
            "jet_definition": jet,
            "recommendation": source_record(cfg_name, cfg_path),
            "histogram_file": {**source_record(main_name, main_path), "objects": main_objects},
            "analysis_file": {**source_record(analysis_name, analysis_path), "objects": analysis_objects},
            "initialized": {f"{mode}_{kind}": sorted(names) for (mode, kind), names in tool_states.items()},
            "states": active_states,
            "source_numerics": SOURCE_NUMERICS,
        }
    return corrections, {
        "release": "AnalysisBase 25.2.117", "athena_commit": ATHENA_COMMIT, "mc_type": MC_TYPE,
        "configured_state_contract": source_record("configured state contract", contract_path),
        "initialized_tools": source_record("Path A initialized tools", initialized_path),
        "collections": provenance,
    }
