"""Locate the wheel's assets and load its compiler runtime once."""

from ctypes import CDLL
import json
import os
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE / "data"
LIB = PACKAGE.parent / "correctionlib" / "lib"
PROVIDER = LIB / "libjets.so"
ROOT_LIBRARIES = (
    "libROOT", "libCore.so", "libRIO.so", "libHist.so", "libTree.so",
    "libCling.so", "libMathCore.so", "libPhysics.so", "libcppyy", "libCPyCppyy",
    "libxAOD", "libJetCalibTools", "libJetUncertainties", "libAsgTools",
)
COMPILER_LIBRARIES = ("libgcc_s.so.1", "libstdc++.so.6")
_HANDLES = []


def mapped_libraries():
    libraries = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/") and ".so" in fields[5]:
            libraries.add(Path(fields[5]).resolve())
    return libraries


def compiler_maps(libraries, name):
    return {path for path in libraries if path.name.startswith(name.split(".so")[0])}


def prepare_runtime():
    """Use one shipped libgcc and libstdc++ per interpreter process."""
    if sys.platform != "linux" or sys.version_info[:2] != (3, 14):
        raise RuntimeError("This wheel requires Linux x86-64 and CPython 3.14")
    for name in COMPILER_LIBRARIES:
        bundled = (LIB / name).resolve(strict=True)
        present = compiler_maps(mapped_libraries(), name)
        if present == {bundled}:
            continue
        if present:
            raise RuntimeError(
                f"{name} was already loaded from {sorted(present)}. "
                "Start a fresh process and initialize this wheel before loading C++ extensions."
            )
        _HANDLES.append(CDLL(str(bundled), mode=os.RTLD_NOW | os.RTLD_GLOBAL))


def audit_runtime():
    """Verify the loaded evaluator, provider and native dependency closure."""
    libraries = mapped_libraries()
    if "ROOT" in sys.modules or any(path.name.startswith(ROOT_LIBRARIES) for path in libraries):
        raise RuntimeError("The standalone process loaded a ROOT or AnalysisBase library")
    if any(path.is_relative_to("/cvmfs") for path in libraries):
        raise RuntimeError("A native library came from CVMFS")
    for name in COMPILER_LIBRARIES + ("libcorrectionlib.so", "libjets.so"):
        expected = (LIB / name).resolve(strict=True)
        observed = compiler_maps(libraries, name)
        if observed != {expected}:
            raise RuntimeError(f"Expected only {expected} for {name}; observed {sorted(observed)}")
    result = sorted(str(path) for path in libraries)
    audit_path = os.environ.get("ATLAS_JET_LIBRARY_AUDIT")
    if audit_path:
        with Path(audit_path).open("x") as stream:
            json.dump({"libraries": result, "python": sys.version}, stream, indent=2)
            stream.write("\n")
    return result
