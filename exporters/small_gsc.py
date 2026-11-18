"""Export the classic trackWIDTH-depth small-R EMPFlow GSC."""
from exporters.calibdata import read_config, resolve, source_record, versioned_calibration_name
from exporters.rootdata import export_histogram


ATHENA_COMMIT = "6ce3f7ae7c41dd9eaec170c3c349df15faef7e5c"
CALIB_AREA = "00-04-83"
CONFIG_NAME = "AntiKt4EMPFlow_MC23a_PreRecR22_Phase2_CalibConfig_ResPU_EtaJES_GSC_241208_InSitu.config"
JET_DEFINITION = "AntiKt4EMPFlow"
NODE_TYPE = "org.atlas.jets.small_gsc"
# GlobalSequentialCorrection checks family names in this order while loading keys.
LOAD_ORDER = ("EM3", "nTrk", "Tile0", "chargedFraction", "trackWIDTH")
APPLY_ORDER = ("chargedFraction", "Tile0", "EM3", "nTrk", "trackWIDTH")


def read_gsc_histograms(root, path, caps):
    response = {name: [] for name in APPLY_ORDER}
    provenance = {name: [] for name in APPLY_ORDER}
    root_file = root.TFile.Open(str(path), "READ")
    if not root_file or root_file.IsZombie():
        raise OSError(f"cannot read GSC response file {path}")
    try:
        previous = None
        for key in root_file.GetListOfKeys():
            name = key.GetName()
            duplicate = previous is not None and previous in name
            previous = name
            if JET_DEFINITION not in name or duplicate:
                continue
            for family in LOAD_ORDER:
                if family not in name or len(response[family]) >= caps[family]:
                    continue
                histogram = root_file.Get(name)
                if not histogram or histogram.ClassName() != "TH2F":
                    raise ValueError(f"expected TH2F for GSC key {name}")
                for axis in (histogram.GetXaxis(), histogram.GetYaxis()):
                    if axis.GetFirst() != 1 or axis.GetLast() != axis.GetNbins():
                        raise ValueError(f"GSC v1 requires an unrestricted histogram axis in {name}")
                response[family].append(export_histogram(histogram))
                provenance[family].append(
                    {"name": name, "cycle": int(root_file.GetKey(name).GetCycle()),
                     "class": histogram.ClassName(), "eta_index": len(response[family]) - 1}
                )
                break
    finally:
        root_file.Close()
    for family, histograms in response.items():
        if len(histograms) < 3:
            raise ValueError(f"GSC expects at least 3 histograms for {family}")
    return response, provenance


def export_small_gsc(area=CALIB_AREA, config_name=CONFIG_NAME):
    """Return the schema-v3 correction and the ordered calibration provenance."""
    import ROOT
    from PathResolver import PathResolver

    global_name = versioned_calibration_name(f"JetCalibTools/CalibrationConfigs/{config_name}", area)
    global_path = resolve(PathResolver, global_name)
    environment = ROOT.TEnv()
    read_config(environment, global_path, ROOT.kEnvGlobal)
    sources = [source_record(global_name, global_path)]

    # JetCalibrationTool gives each step the same TEnv. Initialization of
    # residual pile-up and then EtaJES loads their local files before GSC.
    for key in ("ResidualOffset.CalibFile", "AbsoluteJES.CalibFile"):
        request = versioned_calibration_name(str(environment.GetValue(key, "")), area)
        path = resolve(PathResolver, request)
        read_config(environment, path, ROOT.kEnvLocal)
        sources.append(source_record(request, path))

    settings = {
        "caloBasedGSC": bool(environment.GetValue("caloBasedGSC", False)),
        "GSCDepth": str(environment.GetValue("GSCDepth", "Full")),
        "GSCStartingScale": str(environment.GetValue("GSCStartingScale", "JetEtaJESScaleMomentum")),
        "TurnOffTrackCorrections": bool(environment.GetValue("TurnOffTrackCorrections", False)),
        "nTrkwTrk4PFlow": bool(environment.GetValue("nTrkwTrk4PFlow", False)),
    }
    required = {
        "caloBasedGSC": False, "GSCDepth": "trackWIDTH",
        "GSCStartingScale": "JetEtaJESScaleMomentum",
        "TurnOffTrackCorrections": False, "nTrkwTrk4PFlow": True,
    }
    if settings != required:
        raise ValueError(f"unsupported small-R GSC settings {settings!r}")

    caps = {
        "EM3": int(environment.GetValue("EM3MaxEtaBin", 35)),
        "nTrk": 25, "Tile0": 17, "chargedFraction": 27, "trackWIDTH": 25,
    }
    hist_name = versioned_calibration_name(str(environment.GetValue("GSCFactorsFile", "")), area)
    hist_path = resolve(PathResolver, hist_name)
    response, root_keys = read_gsc_histograms(ROOT, hist_path, caps)

    inputs = [
        ("pt", "real", "EtaJES-scale pT in MeV"),
        ("eta", "real", "EtaJES-scale pseudorapidity"),
        ("phi", "real", "EtaJES-scale azimuth"),
        ("m", "real", "EtaJES-scale mass in MeV"),
        ("detector_eta", "real", "Original float DetectorEta"),
        ("constituent_pt", "real", "Constituent-scale pT in MeV"),
        ("constituent_eta", "real", "Constituent-scale pseudorapidity"),
        ("constituent_phi", "real", "Constituent-scale azimuth"),
        ("constituent_m", "real", "Constituent-scale mass in MeV"),
        ("sum_pt_charged_pfo", "real", "SumPtChargedPFOPt500 at the GSC vertex index"),
        ("n_trk", "int", "NumTrkPt1000 at the same vertex index"),
        ("track_width", "real", "TrackWidthPt1000 at the same vertex index"),
        ("em_b3", "real", "EnergyPerSampling[3] in MeV"),
        ("em_e3", "real", "EnergyPerSampling[7] in MeV"),
        ("tile_bar0", "real", "EnergyPerSampling[12] in MeV"),
        ("tile_ext0", "real", "EnergyPerSampling[18] in MeV"),
    ]
    correction = {
        "name": "small_gsc",
        "version": 1,
        "description": "Run 3 MC small-R EMPFlow global sequential response through trackWIDTH.",
        "inputs": [{"name": name, "type": kind, "description": description} for name, kind, description in inputs],
        "output": {"name": "p4", "type": "real", "shape": [4], "description": "[pt, eta, phi, m]"},
        "data": {
            "nodetype": "extension", "type": NODE_TYPE, "version": 1, "output_shape": [4],
            "config": {
                "abs_eta_width": 0.1, "mev_per_gev": 1000.0, "axis_edge_offset": 1e-6,
                "responses": response,
            },
        },
    }
    provenance = {
        "analysisbase_release": "25.2.117",
        "athena_commit": ATHENA_COMMIT,
        "jet_definition": JET_DEFINITION,
        "calibration_area": area,
        "calibration_sequence": "JetArea_Residual_EtaJES_GSC",
        "sources_in_tenv_resolution_order": sources,
        "response_root_file": source_record(hist_name, hist_path),
        "histogram_load_order": LOAD_ORDER,
        "response_order": APPLY_ORDER,
        "response_caps": caps,
        "response_keys": root_keys,
        "resolved_settings": settings,
        "output_scale": "JetGSCScaleMomentum",
        "event_inputs": {
            "moment_vertex_index": "Use EventInfo.PVIndex when available, else the first PriVtx index or zero. Invalid nonzero index resets to zero when PrimaryVertices is present. A missing PrimaryVertices container returns before index validation.",
            "missing_event_info": "Original calibrator uses the default vertex index zero.",
        },
        "source_contract": {
            "gather": "Require all three per-vertex inputs and four sampling slots before native evaluation. Preserve the checked index and report missing data or out-of-range array accesses.",
            "active_modes": "Run charged fraction, Tile0, EM3, track count, and track width. The chosen depth and properties disable punch-through, track fade-out, and calo-only moments.",
            "response": "A missing per-family eta bin or a nonphysical moment returns unity. Tile0 treats zero as a valid moment. Look up subsequent factors at updated pT and clamp each coordinate 1e-6 into its first/last histogram bins.",
            "precision": "Use binary32 input moments and sampling-fraction arithmetic, binary64 response and cumulative scale, and binary32 p4 writes.",
        },
        "acceptance": "Compare [pt, eta, phi, m] to JetGSCScaleMomentum with terminal eta-family cutoffs, zero/negative moments, and exact TH2 bin-boundary probes.",
    }
    return correction, provenance
