"""Pinned source edits for the classic NNJvt framework extensions."""
from functools import partial
from pathlib import Path


PACKAGE_PREFIX = "Reconstruction/Jet"
RUNTIME_INCLUDE = '#include "JetsCorrectionRuntime/Runtime.h"\n#include <stdexcept>\n\n'
TEMPLATE_DIR = Path(__file__).parent / "jvt"


def replace_one(source, anchor, replacement):
    if source.count(anchor) != 1:
        raise ValueError(f"expected one occurrence of {anchor!r}")
    return source.replace(anchor, replacement, 1)


def include_runtime(source):
    first = source.index("#include")
    return source[:first] + RUNTIME_INCLUDE + source[first:]


def replace_body(source, signature, body, closing_indent):
    if source.count(signature) != 1:
        raise ValueError(f"expected one occurrence of {signature!r}")
    start = source.index(signature) + len(signature)
    end = source.index(f"\n{closing_indent}}}\n", start)
    return source[:start] + "\n" + body.rstrip() + source[end:]


def initialize_gate(source, signature, condition):
    return replace_one(source, signature, signature + f"""
        if ({condition}) {{
            ATH_MSG_ERROR("Unsupported classic NNJvt configuration for the CorrectionLib overlay");
            return StatusCode::FAILURE;
        }}
""")


def patch_update(source):
    signature = "    StatusCode JetVertexNNTagger::decorate(const xAOD::JetContainer& jetCont) const"
    source = replace_body(source, signature + "\n    {", (TEMPLATE_DIR / "update.inc").read_text(), "    ")
    condition = """m_useTrkAugNN || m_maxpt_for_cut != 60000.f ||
            m_NNConfigDir.value() != "JetPileupTag/NNJvt/2022-03-22" ||
            m_NNParamFileName.value() != "NNJVT.Network.graph.Offline.Nonprompt_All_MaxWeight.json" ||
            m_NNCutFileName.value() != "NNJVT.Cuts.FixedEffPt.Offline.Nonprompt_All_MaxW.json" """
    source = initialize_gate(source, "    StatusCode JetVertexNNTagger::initialize()\n    {", condition.strip())
    return include_runtime(source)


def patch_selection(source):
    signature = "    bool NNJvtSelectionTool::select(const xAOD::IParticle *jet) const {"
    source = replace_body(source, signature, (TEMPLATE_DIR / "select.inc").read_text(), "    ")
    condition = """m_wp.value() != "FixedEffPt" ||
            m_configDir.value() != "JetPileupTag/NNJvt/2022-03-22" ||
            (!m_configFile.empty() &&
             m_configFile.value() != "NNJVT.Cuts.FixedEffPt.Offline.Nonprompt_All_MaxW.json") ||
            m_jetEtaName.value() != "eta" || m_minEta != -1.f || m_maxEta != 2.5f ||
            m_minPtForJvt != 20000.f || m_maxPtForJvt != 60000.f"""
    source = initialize_gate(source, "    StatusCode NNJvtSelectionTool::initialize() {", condition)
    marker = "        ATH_CHECK(JvtSelectionToolBase::initialize());"
    source = replace_one(source, marker, marker + """
        if (m_jvtMomentKey.key() != "NNJvt" && !m_jvtMomentKey.key().ends_with(".NNJvt")) {
            ATH_MSG_ERROR("CorrectionLib NNJvt selection requires NNJvt");
            return StatusCode::FAILURE;
        }""")
    return include_runtime(source)


def patch_weight_header(source):
    marker = "        virtual StatusCode sysApplySystematicVariation(const CP::SystematicSet &sys) override;"
    source = replace_one(source, marker, marker + """

        CorrectionCode getEfficiencyScaleFactor(const xAOD::Jet& jet, float& sf) const override;
        CorrectionCode getInefficiencyScaleFactor(const xAOD::Jet& jet, float& sf) const override;""")
    marker = "    private:"
    return replace_one(source, marker, marker + """
        CorrectionCode correctionlibScaleFactor(const xAOD::Jet& jet, float& sf, bool passed) const;
""")


def patch_weight(source):
    condition = """m_wp.value() != "FixedEffPt" || !m_doTruthRequirement ||
            m_truthHSLabel.key() != "isJvtHS" || m_jetEtaName.value() != "eta" ||
            m_minEta != -1.f || m_maxEta != 2.5f || m_minPtForJvt != 20000.f ||
            m_maxPtForJvt != 60000.f ||
            m_file.value() != "JetJvtEfficiency/May2024/NNJvtSFFile_Run3_EMPFlow.root" """
    source = initialize_gate(source, "    StatusCode NNJvtEfficiencyTool::initialize() {", condition.strip())
    marker = "    StatusCode NNJvtEfficiencyTool::sysApplySystematicVariation"
    methods = (TEMPLATE_DIR / "weight.inc").read_text()
    return include_runtime(replace_one(source, marker, methods + marker))


def patch_link(source, token):
    return replace_one(source, token, token + " JetsCorrectionRuntime")


def overrides():
    prefix = f"{PACKAGE_PREFIX}/JetMomentTools"
    efficiency = f"{PACKAGE_PREFIX}/JetJvtEfficiency"
    return {
        f"{prefix}/CMakeLists.txt": partial(
            patch_link, token="PRIVATE_LINK_LIBRARIES CaloGeoHelpers FourMomUtils xAODPFlow CxxUtils PathResolver nlohmann_json::nlohmann_json",
        ),
        f"{prefix}/Root/JetVertexNNTagger.cxx": patch_update,
        f"{efficiency}/CMakeLists.txt": partial(patch_link, token="PRIVATE_LINK_LIBRARIES PathResolver"),
        f"{efficiency}/Root/NNJvtSelectionTool.cxx": patch_selection,
        f"{efficiency}/Root/NNJvtEfficiencyTool.cxx": patch_weight,
        f"{efficiency}/JetJvtEfficiency/NNJvtEfficiencyTool.h": patch_weight_header,
    }
