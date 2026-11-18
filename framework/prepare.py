"""Prepare a WorkDir overlay from the installed release and pinned CERN source."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


COMMIT = "6ce3f7ae7c41dd9eaec170c3c349df15faef7e5c"
UPSTREAM = f"https://gitlab.cern.ch/atlas/athena/-/raw/{COMMIT}"
PACKAGE_NAME = "Reconstruction/Jet/JetCalibTools"
WORKDIR_FILES = (
    "Projects/WorkDir/CMakeLists.txt",
    "Projects/WorkDir/cmake/ActsOverride.cmake",
    "Projects/WorkDir/cmake/VecMemOverride.cmake",
)
NOMINAL_METHODS = {
    "Root/JetPileupCorrection.cxx": (
        "StatusCode JetPileupCorrection::calibrate(xAOD::Jet& jet, JetEventInfo& jetEventInfo) const {",
        "apply_small_pileup(jet, jetEventInfo)",
    ),
    "Root/EtaJESCorrection.cxx": (
        "StatusCode EtaJESCorrection::calibrate(xAOD::Jet& jet, JetEventInfo&) const {",
        "apply_eta_jes(jet, m_jetAlgo.Data(), m_jetStartScale)",
    ),
    "Root/JMSCorrection.cxx": (
        "StatusCode JMSCorrection::calibrate(xAOD::Jet& jet, JetEventInfo&) const {",
        "apply_large_jms(jet)",
    ),
    "Root/GlobalSequentialCorrection.cxx": (
        "StatusCode GlobalSequentialCorrection::calibrate(xAOD::Jet& jet, JetEventInfo& jetEventInfo) const {",
        "apply_small_gsc(jet, jetEventInfo)",
    ),
}
BRIDGE_INCLUDE = '#include "JetsNominalBridge.h"\n\n'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def download(relative):
    return subprocess.check_output(
        ["curl", "--fail", "--silent", "--show-error", "--retry", "2", "--max-time", "30",
         f"{UPSTREAM}/{relative}"]
    )


def unique_index(source, token):
    if source.count(token) != 1:
        raise ValueError(f"expected one occurrence of {token!r}")
    return source.index(token)


def include_bridge(source):
    first_include = source.index("#include")
    return source[:first_include] + BRIDGE_INCLUDE + source[first_include:]


def replace_method(source, signature, call):
    start = unique_index(source, signature) + len(signature)
    end = source.index("\n}\n", start)
    implementation = (
        "\n  try {\n"
        f"    jets::framework::{call};\n"
        "    return StatusCode::SUCCESS;\n"
        "  } catch (const std::exception& error) {\n"
        '    ATH_MSG_ERROR("CorrectionLib nominal stage failed: " << error.what());\n'
        "    return StatusCode::FAILURE;\n"
        "  }"
    )
    return include_bridge(source[:start] + implementation + source[end:])


def validate_tool_recipe(source):
    signature = "StatusCode JetCalibrationTool::initialize() {"
    start = unique_index(source, signature) + len(signature)
    validation = """
  try {
    const bool standard =
        !m_isData && !m_devMode && !m_useOriginVertex && m_gscDepth == "auto" &&
        m_dir == "JetCalibTools/CalibrationConfigs/" && m_forceCampaign.empty() &&
        m_forceCalibFile_PtResidual.empty() && m_forceCalibFile_FastSim.empty() &&
        m_forceCalibFile_MC2MC.empty();
    jets::framework::validate_nominal(m_jetAlgo, m_config, m_calibSeq, m_calibAreaTag, standard);
  } catch (const std::exception& error) {
    ATH_MSG_ERROR("CorrectionLib nominal configuration failed: " << error.what());
    return StatusCode::FAILURE;
  }
"""
    return include_bridge(source[:start] + validation + source[start:])


def link_bridge(source):
    token = "PRIVATE_LINK_LIBRARIES xAODMuon PathResolver AthContainers FourMomUtils)"
    unique_index(source, token)
    return source.replace(token, token[:-1] + " JetsCorrectionRuntime )", 1)


def prepare(release, destination, framework):
    work_src = destination / "src"
    work_src.mkdir()
    source_receipts = {}
    for name in WORKDIR_FILES:
        raw = download(name)
        path = work_src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        source_receipts[name] = {"upstream_sha256": digest(raw)}

    source_pkg = release / "src" / PACKAGE_NAME
    overlay_pkg = work_src / PACKAGE_NAME
    shutil.copytree(source_pkg, overlay_pkg)
    shutil.copytree(framework / "runtime", work_src / "JetsCorrectionRuntime")
    for file in (framework / "nominal").iterdir():
        shutil.copyfile(file, overlay_pkg / "Root" / file.name)

    overrides = {
        **{name: lambda text, sig=sig, call=call: replace_method(text, sig, call)
           for name, (sig, call) in NOMINAL_METHODS.items()},
        "Root/JetCalibrationTool.cxx": validate_tool_recipe,
        "CMakeLists.txt": link_bridge,
    }
    diffs = []
    for filename, patch in overrides.items():
        relative = f"{PACKAGE_NAME}/{filename}"
        path = overlay_pkg / filename
        original = path.read_bytes()
        upstream = download(relative)
        if original != upstream:
            raise ValueError(f"installed file differs from pinned Athena source: {relative}")
        original_text = original.decode()
        patched = patch(original_text)
        path.write_text(patched)
        source_receipts[relative] = {"upstream_sha256": digest(upstream), "overlay_sha256": digest(path.read_bytes())}
        diffs.extend(difflib.unified_diff(
            original_text.splitlines(keepends=True), patched.splitlines(keepends=True),
            fromfile=f"a/{relative}", tofile=f"b/{relative}",
        ))

    (destination / "overlay.diff").write_text("".join(diffs))
    (destination / "source-provenance.json").write_text(
        json.dumps({"athena_commit": COMMIT, "analysisbase": str(release), "files": source_receipts}, indent=2) + "\n"
    )


if __name__ == "__main__":
    release, destination, framework = map(Path, sys.argv[1:])
    prepare(release, destination, framework)
