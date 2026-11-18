"""Register a passive diagnostic through the stock CP text configuration."""

from pathlib import Path

import ROOT

from AnalysisAlgorithmsConfig.ConfigBlock import ConfigBlock


library = Path(__file__).resolve().parents[2] / ".reference-build/libJetsReferenceObserver.so"
if ROOT.gSystem.Load(str(library)) < 0:
    raise RuntimeError(f"Could not load {library}")


class ReferenceObserver(ConfigBlock):
    def makeAlgs(self, config):
        uncertainties = config.getAlgorithm("JetUncertaintiesAlg_SmallR")
        if uncertainties is None:
            raise KeyError("Configured small-R uncertainty algorithm not found")
        uncertainties.outOfValidityDeco = "refJesJerValidity"
        for collection in ("SmallR", "LargeR"):
            alg = config.createAlgorithm("ObserveValidity", f"Observe{collection}")
            alg.jets = config.readName(collection)
            alg.largeR = collection == "LargeR"
