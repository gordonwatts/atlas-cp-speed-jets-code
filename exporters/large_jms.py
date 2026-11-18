"""Export the nominal large-R JMS stage in the pinned AnalysisBase environment."""

import hashlib
from pathlib import Path

from exporters.rootdata import export_histogram

ATHENA_COMMIT = "6ce3f7ae7c41dd9eaec170c3c349df15faef7e5c"
JET_DEFINITION = "AntiKt10UFOCSSKSoftDropBeta100Zcut10"
CALIB_AREA = "00-04-83"
CONFIG_NAME = "JES_MC20PreRecommendation_R10_UFO_CSSK_SoftDrop_JMS_R21Insitu_26Nov2024.config"
NODE_TYPE = "org.atlas.jets.large_jms"


def versioned_calibration_name(requested, area):
    prefix = "JetCalibTools/"
    if not requested.startswith(prefix):
        raise ValueError(f"expected a {prefix} configuration path, got {requested!r}")
    return f"{prefix}CalibArea-{area}/{requested[len(prefix):]}"


def resolve(resolver, requested):
    found = str(resolver.FindCalibFile(requested))
    if not found or not Path(found).is_file():
        raise FileNotFoundError(f"PathResolver could not resolve {requested}")
    return Path(found)


def source_record(requested, path):
    content = path.read_bytes()
    return {
        "path_resolver_name": requested,
        "resolved_path": str(path),
        "real_path": str(path.resolve()),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def read_config(environment, path, level):
    if environment.ReadFile(str(path), level) != 0:
        raise ValueError(f"TEnv could not read {path}")


def extract_jms_settings(environment):
    settings = {
        "JMSStartingScale": str(environment.GetValue("JMSStartingScale", "JetEtaJESScaleMomentum")),
        "JMSOutScale": str(environment.GetValue("JMSOutScale", "JetJMSScaleMomentum")),
        "MinpT": int(environment.GetValue("MinpT", 180)),
        "pTfixed": bool(environment.GetValue("pTfixed", False)),
        "MassCalibrationIs3D": bool(environment.GetValue("MassCalibrationIs3D", False)),
        "JMSBinningParam": str(environment.GetValue("JMSBinningParam", "")),
        "OnlyCombination": bool(environment.GetValue("OnlyCombination", False)),
        "Combination": bool(environment.GetValue("Combination", False)),
        "TrackAssistedJetMassCorr": bool(environment.GetValue("TrackAssistedJetMassCorr", False)),
        "MassCalibrationFile": str(environment.GetValue("MassCalibrationFile", "")),
    }
    required = {
        "JMSStartingScale": "JetEtaJESScaleMomentum",
        "JMSOutScale": "JetJMSScaleMomentum",
        "MassCalibrationIs3D": True,
        "JMSBinningParam": "e_LOGmOe_eta",
        "OnlyCombination": False,
        "Combination": False,
        "TrackAssistedJetMassCorr": False,
    }
    for name, expected in required.items():
        if settings[name] != expected:
            raise ValueError(f"JMS v1 needs {name}={expected!r} but got {settings[name]!r}")
    return settings


def read_jms_histogram(root, path):
    root_file = root.TFile.Open(str(path), "READ")
    if not root_file or root_file.IsZombie():
        raise ValueError(f"cannot read JMS ROOT file {path}")
    try:
        names = {key.GetName() for key in root_file.GetListOfKeys() if JET_DEFINITION in key.GetName()}
        if len(names) != 1:
            raise ValueError(f"expected one JMS object for {JET_DEFINITION}, found {sorted(names)}")
        name = names.pop()
        histogram = root_file.Get(name)
        if not histogram or histogram.ClassName() != "TH3F":
            raise TypeError(f"expected TH3F at {name} in {path}")
        key = root_file.GetKey(name)
        return export_histogram(histogram), {
            "key": name,
            "cycle": int(key.GetCycle()),
            "class": histogram.ClassName(),
            "coordinates": ["energy_GeV", "log_mass_over_energy", "abs_detector_eta"],
        }
    finally:
        root_file.Close()


def correction_def(configuration):
    inputs = [
        ("pt", "EtaJES-stage transverse momentum in MeV"),
        ("eta", "EtaJES-stage pseudorapidity"),
        ("phi", "EtaJES-stage azimuth in radians"),
        ("m", "EtaJES-stage mass in MeV"),
        ("detector_eta", "Original DetectorEta auxiliary"),
    ]
    return {
        "name": "large_jms",
        "description": "AnalysisBase 25.2.117 nominal mass calibration for R=1.0 UFO jets",
        "version": 1,
        "inputs": [{"name": name, "type": "real", "description": description}
                   for name, description in inputs],
        "output": {"name": "p4", "type": "real", "shape": [4], "description": "[pt, eta, phi, m]"},
        "data": {"nodetype": "extension", "type": NODE_TYPE, "version": 1,
                 "output_shape": [4], "config": configuration},
    }


def export_large_jms(area=CALIB_AREA, config_name=CONFIG_NAME):
    """Return a schema-v3 correction definition and its offline source provenance."""
    import ROOT
    from PathResolver import PathResolver

    global_name = versioned_calibration_name(f"JetCalibTools/CalibrationConfigs/{config_name}", area)
    global_path = resolve(PathResolver, global_name)
    environment = ROOT.TEnv()
    read_config(environment, global_path, ROOT.kEnvGlobal)

    # EtaJES initialized first mutates the shared TEnv with this local layer.
    constants_name = versioned_calibration_name(str(environment.GetValue("AbsoluteJES.CalibFile", "")), area)
    constants_path = resolve(PathResolver, constants_name)
    read_config(environment, constants_path, ROOT.kEnvLocal)
    settings = extract_jms_settings(environment)

    root_name = versioned_calibration_name(settings["MassCalibrationFile"], area)
    root_path = resolve(PathResolver, root_name)
    histogram, object_info = read_jms_histogram(ROOT, root_path)
    configuration = {
        "histogram": histogram,
        "binning": settings["JMSBinningParam"],
        # The source names this MinpT but compares it to energy for this binning.
        "minimum_energy_gev": settings["MinpT"],
        "keep_pt_fixed": settings["pTfixed"],
        "axis_edge_offset": 1e-6,
        "mev_per_gev": 1000.0,
    }
    provenance = {
        "analysisbase_release": "25.2.117",
        "athena_commit": ATHENA_COMMIT,
        "jet_definition": JET_DEFINITION,
        "calibration_area": area,
        "calibration_sequence": "EtaJES_JMS",
        "resolved_settings": settings,
        "source_constants": {
            "axis_edge_offset": {"value": 1e-6, "source": "JMSCorrection.cxx:307-314"},
            "mev_per_gev": {"value": 1000.0, "source": "JetCalibrationStep.cxx:13"},
        },
        "global_tenv": source_record(global_name, global_path),
        "etajes_local_tenv": source_record(constants_name, constants_path),
        "response_root_file": source_record(root_name, root_path),
        "response_root_object": object_info,
    }
    return correction_def(configuration), provenance
