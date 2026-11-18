"""Export the release's UFO forward-folding mass-resolution recommendation."""

from pathlib import Path

from exporters.calibdata import read_config, source_record
from exporters.large_jms import ATHENA_COMMIT
from exporters.rootdata import export_histogram

CALIB_AREA = "CalibArea-08"
RECOMMENDATION_DIR = "rel22/Summer2025_PreRec"
NODE_TYPE = "org.atlas.jets.large_jmr"

# FFJetSmearingTool.cxx:493-550 and ParticleJetTools/LargeRJetLabelEnum.h.
# Unknown and other unmatched label values use the release's QCD fallback.
TOPOLOGY_BY_LABEL = (
    "QCD", "Top", "V", "QCD", "QCD", "V", "V", "Top", "V", "no_match",
    "QCD", "H", "QCD", "H", "QCD", "QCD", "QCD", "QCD", "QCD", "QCD",
)

SOURCE_CONSTANTS = {
    "max_match_delta_r": {"value": 0.75, "source": "FFJetSmearingTool.cxx:500-508"},
    "mev_to_gev": {"value": 1e-3, "source": "FFJetSmearingTool.h:210", "type": "float"},
    "axis_edge_offset": {"value": 1e-6, "source": "FFJetSmearingTool.cxx:978-984"},
    "minimum_shift": {"value": 1e-4, "source": "FFJetSmearingTool.cxx:827"},
    "nominal_scale": {"value": 1.0, "source": "FFJetSmearingTool.cxx:765"},
    "nominal_resolution": {"value": 1.0, "source": "FFJetSmearingTool.cxx:765"},
    "zero_response_fallback": {"value": 1.0, "source": "FFJetSmearingTool.cxx:760"},
}


def resolve_uncertainty_file(resolver, filename):
    # jet::utils::findFilePath tries a raw local path, the versioned campaign
    # path via PathResolver, and then the bare name via PathResolver.
    raw_path = Path(filename)
    if raw_path.is_file():
        return filename, raw_path
    versioned_name = f"JetUncertainties/{CALIB_AREA}/{filename}"
    for requested in (versioned_name, filename):
        result = str(resolver.FindCalibFile(requested))
        if result and Path(result).is_file():
            return requested, Path(result)
    raise FileNotFoundError(f"PathResolver could not resolve the JMR input {filename!r}")


def read_histogram(root_file, object_path, dimension):
    directory_path, _, name = object_path.rpartition("/")
    directory = root_file.GetDirectory(directory_path) if directory_path else root_file
    if not directory:
        raise ValueError(f"missing JMR ROOT directory for {object_path}")
    histogram = directory.Get(name)
    if (not histogram or not histogram.InheritsFrom(f"TH{dimension}") or
            (dimension == 3 and histogram.ClassName() != "TH3F")):
        raise TypeError(f"unexpected large-R histogram at {object_path}")
    return export_histogram(histogram), {
        "key": object_path,
        "cycle": int(directory.GetKey(name).GetCycle()),
        "class": histogram.ClassName(),
    }


def extract_components(environment, root_file, mc_type):
    definitions, provenance = [], []
    for index in range(999):
        prefix = f"JMRComponent.{index}."
        if environment.GetValue(f"JMSComponent.{index}.Name", ""):
            raise ValueError("the configured JMR node supports no FF mass-scale components")
        raw_name = str(environment.GetValue(prefix + "Name", ""))
        if not raw_name:
            continue

        name = raw_name.replace("MCTYPE", mc_type)
        binning = str(environment.GetValue(prefix + "Param", ""))
        mass_definition = str(environment.GetValue(prefix + "MassDef", ""))
        topology = str(environment.GetValue(prefix + "Topology", ""))
        if binning not in ("PtAbsMass", "eLOGmOeAbsEta") or mass_definition != "UFO":
            raise ValueError(f"unsupported JMR component {name}, {mass_definition}, {binning}")
        path = str(environment.GetValue(prefix + "Hist", "")).replace("MCTYPE", mc_type)
        dimension = 2 if binning == "PtAbsMass" else 3
        histogram, object_info = read_histogram(root_file, path, dimension)
        states = {f"{name}__1up": 1, f"{name}__1down": -1}
        definitions.append({
            "binning": binning, "topology": topology, "histogram": histogram, "states": states,
        })
        coordinates = (["pt_GeV_float", "mass_GeV_float"] if dimension == 2
                       else ["energy_GeV", "log_mass_over_energy", "abs_eta"])
        provenance.append({
            "config_prefix": prefix, "name": name, "raw_name": raw_name, "states": states,
            "mass_definition": mass_definition, "topology": topology,
            "parameterization": binning, "coordinates": coordinates, "histogram": object_info,
        })
    if not definitions:
        raise ValueError("the configured JMR recommendation has no components")
    return definitions, provenance


def correction_def(configuration):
    inputs = [
        ("pt", "real", "Post-calibration transverse momentum in MeV"),
        ("eta", "real", "Post-calibration pseudorapidity"),
        ("phi", "real", "Post-calibration azimuth in radians"),
        ("m", "real", "Post-calibration mass in MeV"),
        ("truth_label", "int", "Original R10TruthLabel_R21Precision_2022v1 value"),
        ("truth_label_available", "int", "One when the original truth-label decoration exists"),
        ("systematic", "string", "Configured FF systematic name, or a propagated external state"),
        ("event_index", "int", "Zero-based event index in the bound truth-jet batch"),
    ]
    return {
        "name": "large_jmr",
        "description": "AnalysisBase 25.2.117 UFO forward-folding mass resolution",
        "version": 1,
        "inputs": [{"name": name, "type": kind, "description": description}
                   for name, kind, description in inputs],
        "output": {"name": "corrected_jet", "type": "real", "shape": [6],
                   "description": "[pt, eta, phi, m, CorrectionCode, status_available]"},
        "data": {"nodetype": "extension", "type": NODE_TYPE, "version": 1,
                 "output_shape": [6], "config": configuration},
    }


def export_large_jmr(model="Full", mc_type="MC23"):
    """Return the schema-v3 correction definition and offline source provenance."""
    import ROOT
    from PathResolver import PathResolver

    if model not in ("Full", "Simple"):
        raise ValueError(f"unsupported JMR recommendation {model!r}")
    config_name, config_path = resolve_uncertainty_file(
        PathResolver, f"{RECOMMENDATION_DIR}/R10_{model}JMR.config")
    environment = ROOT.TEnv()
    read_config(environment, config_path, ROOT.kEnvGlobal)
    supported_mc_types = {item.strip() for item in str(environment.GetValue("SupportedMCTypes", "")).split(",")}
    if not mc_type or mc_type not in supported_mc_types:
        raise ValueError(f"unsupported JMR MC type {mc_type!r}")
    gaussian = bool(environment.GetValue("doGaussianSmearing", True))
    if gaussian:
        raise ValueError("the configured JMR node requires the UFO response map")

    eta_range = int(environment.GetValue("EtaRange", 0))
    max_mass = int(environment.GetValue("MaxMass", 0))
    max_pt = int(environment.GetValue("MaxPt", 0))
    truth_collection = str(environment.GetValue("TruthJetColl", ""))
    truth_label = str(environment.GetValue("TruthLabelAccessor", ""))
    if (min(eta_range, max_mass, max_pt) <= 0 or
            truth_collection != "AntiKt10TruthSoftDropBeta100Zcut10Jets" or
            truth_label != "R10TruthLabel_R21Precision_2022v1"):
        raise ValueError("invalid ranges or unsupported truth inputs for the configured JMR node")

    uncertainty_root = str(environment.GetValue("UncertaintyRootFile", ""))
    root_name, root_path = resolve_uncertainty_file(PathResolver, uncertainty_root)
    root_file = ROOT.TFile.Open(str(root_path), "READ")
    if not root_file or root_file.IsZombie():
        raise ValueError(f"cannot read the JMR ROOT file {root_path}")
    try:
        response_name = str(environment.GetValue("UFOResponseMap", ""))
        response, response_info = read_histogram(root_file, response_name, 2)
        components, component_info = extract_components(environment, root_file, mc_type)
    finally:
        root_file.Close()

    configuration = {
        "response": response, "components": components,
        "max_abs_eta": eta_range, "max_mass_mev": max_mass, "max_pt_mev": max_pt,
        "fallback_topology": "QCD", "topology_by_label": list(TOPOLOGY_BY_LABEL),
        **{name: detail["value"] for name, detail in SOURCE_CONSTANTS.items()},
    }
    provenance = {
        "analysisbase_release": "25.2.117",
        "athena_commit": ATHENA_COMMIT,
        "mc_type": mc_type,
        "model": model,
        "mass_definition": "UFO",
        "resolved_settings": {
            "UncertaintyRelease": str(environment.GetValue("UncertaintyRelease", "")),
            "EtaRange": eta_range, "MaxMass": max_mass, "MaxPt": max_pt,
            "TruthJetColl": truth_collection, "TruthLabelAccessor": truth_label,
            "UncertaintyRootFile": uncertainty_root, "UFOResponseMap": response_name,
            "doGaussianSmearing": gaussian,
        },
        "source_constants": SOURCE_CONSTANTS,
        "truth_topology_source": "ParticleJetTools/LargeRJetLabelEnum.h and FFJetSmearingTool.cxx:525-549",
        "config_tenv": source_record(config_name, config_path),
        "uncertainty_root_file": source_record(root_name, root_path),
        "response_root_object": {
            **response_info, "coordinates": ["reco_pt_GeV", "matched_truth_mass_GeV"],
        },
        "components": component_info,
    }
    return correction_def(configuration), provenance
