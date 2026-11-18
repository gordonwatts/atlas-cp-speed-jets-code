"""Export the spline-based EtaJES prescriptions used for both Run 3 collections."""

from exporters.calibdata import read_config, resolve, source_record, versioned_calibration_name
from exporters.rootdata import export_histogram

CONFIGS = {
    "small": (
        "AntiKt4EMPFlow",
        "AntiKt4EMPFlow_MC23a_PreRecR22_Phase2_CalibConfig_ResPU_EtaJES_GSC_241208_InSitu.config",
    ),
    "large": (
        "AntiKt10UFOCSSKSoftDropBeta100Zcut10",
        "JES_MC20PreRecommendation_R10_UFO_CSSK_SoftDrop_JMS_R21Insitu_26Nov2024.config",
    ),
}


def export_eta_jes(collection):
    import ROOT
    from PathResolver import PathResolver

    definition, config_name = CONFIGS[collection]
    env = ROOT.TEnv()
    sources = []

    def locate(name):
        logical = versioned_calibration_name(name, "00-04-83")
        path = resolve(PathResolver, logical)
        sources.append(source_record(logical, path))
        return path

    def numbers(name):
        return [float(v) for v in str(env.GetValue(name, "")).split()]

    read_config(env, locate(f"JetCalibTools/CalibrationConfigs/{config_name}"), ROOT.kEnvGlobal)
    if collection == "small":
        read_config(env, locate(str(env.GetValue("ResidualOffset.CalibFile", ""))), ROOT.kEnvLocal)
    read_config(env, locate(str(env.GetValue("AbsoluteJES.CalibFile", ""))), ROOT.kEnvLocal)
    # Match TEnv overloads and defaults in the source, including int min-pt.
    if not env.GetValue(definition + ".isSpline", False):
        raise ValueError("only the configured histogram response is supported")
    if env.GetValue("LowPtJESExtrapolationMethod", 0) != 1 or env.GetValue("ApplyMassCorrection", False):
        raise ValueError("EtaJES v1 requires linear low-pT extension without an internal mass scale")
    eta_edges = numbers("JES.EtaBins") or [0.1 * i - 4.5 for i in range(91)]
    use_secondary = bool(env.GetValue(definition + ".UseSecondaryMinPtForETAJES", False))
    freeze = bool(env.GetValue(definition + ".FreezeJEScorrectionatHighE", False))
    root_path = locate(str(env.GetValue("AbsoluteJES.CalibHists", "")))
    with ROOT.TFile.Open(str(root_path)) as src:
        tables = src.Get("etaJes")
        if not tables or tables.GetSize() < len(eta_edges) - 1:
            raise ValueError("missing per-eta response tables")
        bins = []
        names = []
        for i in range(len(eta_edges) - 1):
            hist = tables.At(i)
            if not hist or hist.GetDimension() != 1:
                raise ValueError(f"expected 1D response for eta bin {i}")
            table = export_histogram(hist)
            # The source explicitly clears Sumw2 on the runtime response tables.
            table["sumw2"] = []
            coeff = numbers(f"EtaCorr.{definition}_Bin{i}")
            if not 1 <= len(coeff) <= 9:
                raise ValueError("invalid eta correction coefficient count")
            freeze_at = numbers(f"EmaxJES.{definition}_Bin{i}") if freeze else [-1.0]
            if len(freeze_at) != 1:
                raise ValueError("expected a single high-energy boundary")
            bins.append({"response": table, "eta_coefficients": coeff, "freeze_at": freeze_at[0]})
            names.append(hist.GetName())
    if len({len(b["eta_coefficients"]) for b in bins}) != 1:
        raise ValueError("non-uniform coefficient counts require an explicit source-order review")
    settings = {
        "eta_edges": eta_edges, "bins": bins,
        "min_pt_gev": int(env.GetValue(definition + ".MinPtForETAJES", 10)),
        "use_secondary": use_secondary,
        "secondary_abs_eta": float(env.GetValue(definition + ".EtaSecondaryMinPtForETAJES", 1.9)),
        "secondary_pt_gev": int(env.GetValue(definition + ".SecondaryMinPtForETAJES", 7)),
        "eta_min_pt_gev": 8.0, "eta_max_energy_gev": 2500.0, "mev_per_gev": 1000.0,
    }
    correction = {
        "name": f"{collection}_eta_jes", "version": 1,
        "inputs": [{"name": n, "type": "real"} for n in ["pt","eta","phi","m","detector_eta"]],
        "output": {"name": "p4", "type": "real", "shape": [4]},
        "data": {"nodetype": "extension", "type": "org.atlas.jets.eta_jes", "version": 1,
                 "output_shape": [4], "config": settings},
    }
    origin = {"jet_definition": definition, "sources_in_resolution_order": sources,
              "response_list": "etaJes", "ordered_histograms": names,
              "starting_scale": str(env.GetValue("EtaJESStartingScale", "JetPileupScaleMomentum")),
              "source": "Reconstruction/Jet/JetCalibTools/Root/EtaJESCorrection.cxx",
              "active_branches": {"isSpline": True, "lowPtExtrap": 1, "applyMassCorrection": False,
                                  "freeze_at_high_energy": freeze}}
    return correction, origin
