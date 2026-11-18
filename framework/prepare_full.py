"""Pinned source edits for event context and non-JVT jet uncertainties."""
from functools import partial
from pathlib import Path

from prepare_jvt import replace_body, replace_one


FRAGMENTS = Path(__file__).parent / "full"
UNCERTAINTIES = "Reconstruction/Jet/JetUncertainties"
LABELING = "PhysicsAnalysis/AnalysisCommon/ParticleJetTools"
ANALYSIS = "PhysicsAnalysis/Algorithms/AsgAnalysisAlgorithms"
CALIBRATION = "Reconstruction/Jet/JetCalibTools"
JET_MOMENTS = "Reconstruction/Jet/JetMomentTools"
FULL_PACKAGES = (UNCERTAINTIES, LABELING, ANALYSIS)
RUNTIME = '#include "JetsCorrectionRuntime/Runtime.h"\n'
EVENTS = '#include "JetsCorrectionRuntime/Events.h"\n'
JET_IO = '#include "JetsCorrectionRuntime/JetIO.h"\n'


def fragment(name):
    return (FRAGMENTS / f"{name}.inc").read_text().rstrip()


def include_runtime(source, header):
    first = source.index("#include")
    return source[:first] + header + "#include <stdexcept>\n\n" + source[first:]


def insert_after(source, anchor, body):
    return replace_one(source, anchor, anchor + "\n" + body.rstrip())


def patch_link(source, token):
    return replace_one(source, token, token + " JetsCorrectionRuntime")


def patch_uncertainties(source):
    signature = "CP::CorrectionCode JetUncertaintiesTool::applyCorrection(xAOD::Jet& jet, const xAOD::EventInfo& eInfo) const"
    source = replace_body(source, signature + "\n{", fragment("uncertainty"), "")
    signature = "StatusCode JetUncertaintiesTool::initialize()\n{"
    source = insert_after(source, signature, """
    const bool small = m_jetDef == "AntiKt4EMPFlow" &&
        m_configFile == "rel22/Winter2025_PreRec/R4_CategoryReduction_FullJER.config";
    const bool large = m_jetDef == "AntiKt10UFOCSSKSoftDropBeta100Zcut10" &&
        m_configFile == "rel22/Summer2025_PreRec/R10_CategoryJES_FullJER_FullJMS.config";
    if ((!small && !large) || m_mcType != "MC23" || m_calibArea != "CalibArea-08" ||
        m_isData != m_pseudoDataJERsmearingMode || !m_path.empty() ||
        !m_analysisFile.empty() || !m_systFilters.empty()) {
      ATH_MSG_ERROR("Unsupported CorrectionLib jet uncertainty configuration");
      return StatusCode::FAILURE;
    }""")
    original = """    unsigned NPV = 0;
    xAOD::VertexContainer::const_iterator itr;
    for (itr = vertices->begin(); itr != vertices->end(); ++itr)
        if ( (*itr)->nTrackParticles() > 1)
            NPV++;

    // Add NPV to the EventInfo object as a decoration.
    decNPV(*ei) = NPV;"""
    replacement = """    try {
        const auto result = jets::framework::event_inputs(*vertices, 0, false, 0., false);
        decNPV(*ei) = static_cast<float>(result[1]);
    } catch (const std::exception& error) {
        ATH_MSG_ERROR("CorrectionLib uncertainty event inputs failed: " << error.what());
        return nullptr;
    }"""
    return include_runtime(replace_one(source, original, replacement), EVENTS + JET_IO)


def patch_ff(source):
    signature = "    CP::CorrectionCode FFJetSmearingTool::applyCorrection( xAOD::Jet& jet_reco) const{"
    source = replace_body(source, signature, fragment("ff"), "    ")
    signature = "    StatusCode FFJetSmearingTool::initialize(){"
    source = insert_after(source, signature, """
        if (m_MassDef_string != "UFO" || m_MCType_string != "MC23" || !m_path.empty() ||
            m_configFile != "rel22/Summer2025_PreRec/R10_FullJMR.config") {
            ATH_MSG_ERROR("Unsupported CorrectionLib forward-folding configuration");
            return StatusCode::FAILURE;
        }""")
    return include_runtime(source, EVENTS + JET_IO)


def patch_pileup_truth(source):
    signature = "StatusCode JetPileupLabelingTool::decorate(const xAOD::JetContainer& jets) const {"
    source = replace_body(source, signature, fragment("pileup_truth"), "")
    source = insert_after(source, "StatusCode JetPileupLabelingTool::initialize(){", """
  if (m_truthJetsKey.key() != "AntiKt4TruthDressedWZJets" || m_hsMaxDR != 0.3f ||
      m_hsMinPt != 10000.f || m_puMinDR != 0.6f || m_puMinPt != 0.f) {
    ATH_MSG_ERROR("Unsupported CorrectionLib pile-up truth label configuration");
    return StatusCode::FAILURE;
  }""")
    return include_runtime(source, EVENTS)


def patch_event_weights(source):
    start = source.index("      float scaleFactor = 1;")
    marker = "      m_scaleFactorOutputDecoration.set (*eventInfo, scaleFactor, sys);"
    end = source.index(marker, start) + len(marker)
    source = source[:start] + fragment("event_sf") + source[end:]
    return include_runtime(source, EVENTS)


def patch_selection(source, signature, name):
    source = insert_after(source, signature, fragment(name))
    return include_runtime(source, RUNTIME)


def patch_nominal_event(source):
    start = source.index("    // If this is GSC, we need EventInfo to determine the PV to use")
    end = source.index("  } else if (m_doDNNCal) {", start)
    source = source[:start] + fragment("nominal_event") + "\n" + source[end:]
    return include_runtime(source, EVENTS)


def patch_jvt_vertex(source):
    original = """        const xAOD::Vertex* hs = findHSVertex(vertices);
        if (!hs) {
            ATH_MSG_ERROR("No primary vertex or single NoVtx sentinel for NNJvt");
            return StatusCode::FAILURE;
        }"""
    source = replace_one(source, original, fragment("jvt_vertex"))
    source = replace_one(source, "        const std::int64_t vertex = hs->vertexType() == xAOD::VxType::PriVtx ? 2 : 1;\n", "")
    source = replace_one(source, "track_sums.at(hs->index() - vertices[0]->index())", "track_sums.at(vertex_slot)")
    return include_runtime(source, EVENTS)


def overrides():
    return {
        f"{UNCERTAINTIES}/CMakeLists.txt": partial(
            patch_link, token="PRIVATE_LINK_LIBRARIES xAODCore xAODTracking PathResolver ParticleJetToolsLib"),
        f"{UNCERTAINTIES}/Root/JetUncertaintiesTool.cxx": patch_uncertainties,
        f"{UNCERTAINTIES}/Root/FFJetSmearingTool.cxx": patch_ff,
        f"{LABELING}/CMakeLists.txt": partial(patch_link, token="PRIVATE_LINK_LIBRARIES AthContainers CxxUtils TruthUtils"),
        f"{LABELING}/Root/JetPileupLabelingTool.cxx": patch_pileup_truth,
        f"{ANALYSIS}/CMakeLists.txt": partial(patch_link, token="xAODTracking xAODTruth xAODRootAccess RootCoreUtils TruthUtils"),
        f"{ANALYSIS}/Root/AsgEventScaleFactorAlg.cxx": patch_event_weights,
        f"{ANALYSIS}/Root/AsgPtEtaSelectionTool.cxx": partial(
            patch_selection, signature="  accept (const xAOD::IParticle *particle) const\n  {",
            name="kinematic_selection"),
        f"{ANALYSIS}/Root/AsgMassSelectionTool.cxx": partial(
            patch_selection, signature="  accept (const xAOD::IParticle *particle) const\n  {",
            name="mass_selection"),
        f"{CALIBRATION}/Root/JetCalibrationTool.cxx": patch_nominal_event,
        f"{JET_MOMENTS}/Root/JetVertexNNTagger.cxx": patch_jvt_vertex,
    }
