"""Independent staged oracle checks for the common EtaJES and large-R JMS nodes."""

import json
from pathlib import Path

import awkward as ak
import correctionlib
from correctionlib import plugins
import numpy as np
import typer
import uproot


def main(original: Path, authority: Path, provider: Path, payload: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    plugins.enable_discovery(False)
    plugins.load_library(provider)
    nodes = correctionlib.CorrectionSet.from_file(str(payload))
    quantities = ("pt", "eta", "phi", "m")
    summary = {"provider": str(provider.resolve()), "payload": str(payload.resolve()),
               "authority": str(authority.resolve()), "original": str(original.resolve()), "checks": {}}
    synchronous = {"handler": uproot.source.file.MemmapSource, "num_fallback_workers": 1,
                   "decompression_executor": uproot.source.futures.TrivialExecutor(),
                   "interpretation_executor": uproot.source.futures.TrivialExecutor()}
    with uproot.open(f"{original}:CollectionTree", **synchronous) as src, uproot.open(f"{authority}:analysis", **synchronous) as ref:
        current = {s.split(":")[0]:s.split(":")[1].strip() for s in Path("/proc/self/status").read_text().splitlines()
                   if s.startswith(("Threads:", "Cpus_allowed_list:"))}
        if current != {"Threads":"1","Cpus_allowed_list":"3"}:
            raise RuntimeError(f"unexpected runtime placement: {current}")
        entries = ref.num_entries
        summary["entries"] = entries
        for key, prefix, collection in (("small", "sr", "AnalysisJets"), ("large", "lr", "AnalysisLargeRJets")):
            def col(branch):
                return src[f"{collection}AuxDyn.{branch}"].array(entry_stop=entries)
            def field(stage, name):
                return ref[f"{prefix}_{stage}{name}_NOSYS"].array()
            def flat(values):
                return np.asarray(ak.flatten(values), dtype=np.float64)
            raw = [col("JetConstitScaleMomentum_" + q) for q in quantities]
            det = col("DetectorEta")
            report = {}
            for i, q in enumerate(quantities):
                assert ak.almost_equal(raw[i], field("const", q), rtol=0, atol=0)
            original_offsets = ak.to_numpy(ak.num(det))
            start = [field("pileup", q) for q in quantities] if key == "small" else raw
            expected_eta = [field("etaJES", q) for q in quantities]
            for array in start + expected_eta:
                np.testing.assert_array_equal(original_offsets, ak.to_numpy(ak.num(array)))
            direct = nodes[key + "_eta_jes"].evaluate(*(flat(v) for v in start), flat(det))
            def compare(name, candidate, reference):
                data = {}
                for i,q in enumerate(quantities):
                    a,b = candidate[:,i], flat(reference[i])
                    mismatch = np.flatnonzero(a != b)
                    data[q] = {"objects": len(b), "exact_mismatches": mismatch.size,
                               "max_abs": float(np.max(np.abs(a-b), initial=0)),
                               "first": None if not mismatch.size else {
                                   "flat_index":int(mismatch[0]), "candidate":float(a[mismatch[0]]),
                                   "A":float(b[mismatch[0]])}}
                report[name] = data
            compare("eta_jes", direct, expected_eta)
            if key == "large":
                expected_jms = [field("jms", q) for q in quantities]
                jms = nodes["large_jms"].evaluate(*(flat(v) for v in expected_eta), flat(det))
                compare("jms_from_A_eta_jes", jms, expected_jms)
                combined = nodes["large_jms"].evaluate(*direct.T, flat(det))
                compare("jms_from_original_constituent_scale", combined, expected_jms)
            summary["checks"][key] = report
    summary["loaded"] = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                                if "/" in line and any(x in line for x in ("libjets.so","libcorrectionlib.so","_core.cpython","libJetCalibToolsLib"))})
    summary["limits"] = {line.split(":")[0]:line.split(":")[1].strip()
                         for line in Path("/proc/self/status").read_text().splitlines()
                         if line.startswith(("Threads:", "Cpus_allowed_list:"))}
    summary["passed"] = summary["limits"] == {"Threads":"1","Cpus_allowed_list":"3"} and all(row["exact_mismatches"] == 0 for size in summary["checks"].values()
                            for stage in size.values() for row in stage.values())
    with output.open("x") as target:
        json.dump(summary,target,indent=2)
        target.write("\n")
    if not summary["passed"]:
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
