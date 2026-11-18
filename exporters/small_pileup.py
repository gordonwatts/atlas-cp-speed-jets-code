"""Export simultaneous small-R area and residual pile-up from the pinned release."""
import math

from exporters.calibdata import read_config, resolve, source_record, versioned_calibration_name


ATHENA_COMMIT = "6ce3f7ae7c41dd9eaec170c3c349df15faef7e5c"
CALIB_AREA = "00-04-83"
CONFIG_NAME = "AntiKt4EMPFlow_MC23a_PreRecR22_Phase2_CalibConfig_ResPU_EtaJES_GSC_241208_InSitu.config"
JET_DEFINITION = "AntiKt4EMPFlow"
NODE_TYPE = "org.atlas.jets.small_pileup"


def export_small_pileup(area=CALIB_AREA, config_name=CONFIG_NAME):
    """Return the schema-v3 correction and the release source receipt."""
    import ROOT
    from PathResolver import PathResolver

    global_name = versioned_calibration_name(f"JetCalibTools/CalibrationConfigs/{config_name}", area)
    global_path = resolve(PathResolver, global_name)
    environment = ROOT.TEnv()
    read_config(environment, global_path, ROOT.kEnvGlobal)

    # The source reads these flags and offset reference values before the
    # residual file is merged into the shared TEnv.
    for key in (
        "ApplyOnlyResidual", "ApplyOnlyMuResidual", "ApplyOnlyNPVResidual",
        "ApplyOnlyNJetResidual", "DoSequentialResidual", "Do3DCorrection",
        "OffsetCorrection.UseNjet", "ApplyFullJetArea4MomentumCorrection",
        "OriginCorrectedClusters",
    ):
        if environment.GetValue(key, False):
            raise ValueError(f"small-R MC pile-up v1 requires {key}=false")
    rho_container = str(environment.GetValue("RhoKey", "None"))
    starting_scale = str(environment.GetValue("PileupStartingScale", "JetConstitScaleMomentum"))
    if rho_container != "Kt4EMPFlowNeutEventShape" or starting_scale != "JetConstitScaleMomentum":
        raise ValueError(f"unsupported pile-up inputs: {rho_container}, {starting_scale}")
    mu_ref = float(environment.GetValue("OffsetCorrection.DefaultMuRef", -99.0))
    npv_ref = float(environment.GetValue("OffsetCorrection.DefaultNPVRef", -99.0))
    mu_factor = float(environment.GetValue("MuScaleFactor", 1.0))
    if -99.0 in (mu_ref, npv_ref) or any(not math.isfinite(x) for x in (mu_ref, npv_ref, mu_factor)):
        raise ValueError("invalid residual reference values")

    residual_name = versioned_calibration_name(str(environment.GetValue("ResidualOffset.CalibFile", "")), area)
    residual_path = resolve(PathResolver, residual_name)
    read_config(environment, residual_path, ROOT.kEnvLocal)
    if environment.GetValue("ApplyNPVBeamspotCorrection", False):
        raise ValueError("small-R MC pile-up v1 requires ApplyNPVBeamspotCorrection=false")
    name = str(environment.GetValue("ResidualOffsetCorrection.Name", ""))

    def parameters(key):
        values = [float(x) for x in str(environment.GetValue(f"{name}.{key}", "")).split()]
        if not values or not all(map(math.isfinite, values)):
            raise ValueError(f"invalid {name}.{key}")
        return values

    edges = parameters("AbsEtaBins")
    mu = parameters(f"MuTerm.{JET_DEFINITION}")
    npv = parameters(f"NPVTerm.{JET_DEFINITION}")
    if len(edges) < 3 or len(mu) != len(edges) or len(npv) != len(edges):
        raise ValueError("each residual coefficient series must contain an intercept and one slope per eta interval")
    if edges[0] != 0.0 or any(low >= high for low, high in zip(edges, edges[1:])):
        raise ValueError("invalid residual absolute-eta edges")

    config = {
        "abs_eta_edges": edges, "mu_coefficients": mu, "npv_coefficients": npv,
        "mu_ref": mu_ref, "npv_ref": npv_ref, "mu_factor": mu_factor,
        "mev_per_gev": 1000.0, "negative_reset_pt_gev": 0.01,
    }
    correction = {
        "name": "small_pileup",
        "version": 1,
        "description": "Run 3 small-R MC constituent-scale area and simultaneous residual pile-up correction.",
        "inputs": [
            {"name": name, "type": kind, "description": description}
            for name, kind, description in (
                ("pt", "real", "Constituent-scale pT in MeV, as stored in the jet"),
                ("eta", "real", "Constituent-scale pseudorapidity"),
                ("phi", "real", "Constituent-scale azimuth"),
                ("m", "real", "Constituent-scale mass in MeV"),
                ("area_pt", "real", "Float ActiveArea4vec_pt"),
                ("rho", "real", "Event Shape Density in MeV per unit area"),
                ("mu", "real", "Float averageInteractionsPerCrossing"),
                ("npv", "int", "Count of PriVtx or PileUp vertices"),
            )
        ],
        "output": {"name": "p4", "type": "real", "shape": [4], "description": "[pt, eta, phi, m]"},
        "data": {
            "nodetype": "extension", "type": NODE_TYPE, "version": 1,
            "output_shape": [4], "config": config,
        },
    }
    provenance = {
        "analysisbase_release": "25.2.117",
        "athena_commit": ATHENA_COMMIT,
        "jet_definition": JET_DEFINITION,
        "calibration_area": area,
        "calibration_sequence": "JetArea_Residual_EtaJES_GSC",
        "resolved_files": [source_record(global_name, global_path), source_record(residual_name, residual_path)],
        "residual_name": name,
        "starting_scale": starting_scale,
        "output_scale": "JetPileupScaleMomentum",
        "rho_container": rho_container,
        "event_inputs": {
            "rho": "Missing density or event shape is an event failure.",
            "mu": "Use float EventInfo.averageInteractionsPerCrossing. Missing EventInfo sets mu and NPV to zero; missing decoration on available EventInfo fails.",
            "npv": "Count PriVtx or PileUp in PrimaryVertices; use zero on missing PrimaryVertices.",
        },
        "source_contract": {
            "cases": "Scalar area subtraction and simultaneous mu/NPV residual on MC.",
            "negative_pt": "Only negative corrected pT resets to 10 MeV through four-vector scaling.",
            "invalid_inputs": "Missing required jet auxiliary values and E<m are errors.",
            "eta_boundary": "The release residual loop indexes past the last coefficient if no interval matches. The node reports that undefined input as an error.",
            "precision": "Retain binary32 auxiliary reads and writes, binary64 density and arithmetic, and the source interval accumulation order.",
        },
        "acceptance": "Compare [pt, eta, phi, m] to JetPileupScaleMomentum and exercise the negative reset, zero corrected pT, and residual eta edges.",
    }
    return correction, provenance
