# ATLAS CP speed study for jets

This repository contains the implementation, validation and benchmark drivers for an independent conversion of the ATLAS Run 3 small-R and large-R jet calibrations to CorrectionLib.

The original campaign history is intact through [0b06f13](https://github.com/gordonwatts/atlas-cp-speed-jets-code/commit/0b06f13f140db0c34c73bfe0676cd429849ff72c). Clone with submodules to obtain the pinned CorrectionLib fork.

```bash
git clone --recurse-submodules https://github.com/gordonwatts/atlas-cp-speed-jets-code.git
```

- [Study report and slides](https://github.com/gordonwatts/atlas-cp-speed-jets)
- [Reproduction instructions](https://github.com/gordonwatts/atlas-cp-speed-jets/blob/main/jets-skill-c8a2374-20261117/reproduce.md)
- [Frozen workload contract](contracts/workload.json)

The report repository includes the wheel and runnable example, figures, and saved evidence. Rebuilding and rerunning the full campaign also requires the release and calibration input described in the reproduction instructions.

## Repository map

| Path | What lives there |
| --- | --- |
| [`config/`](config/) and [`contracts/`](contracts/) | AnalysisBase YAML for the three jet scopes, frozen input and systematic-state lists, histogram definitions, and the workload and shipment specifications. |
| [`tools/reference/`](tools/reference/) | **Path A**. Sets up stock AnalysisBase, runs the reference observer, and fills the comparison histograms. Start with [`workflow.py`](tools/reference/workflow.py) for the histogram logic. |
| [`framework/`](framework/) | **Path B**. AnalysisBase adapters and the runtime bridge to CorrectionLib. [`prepare_full.py`](framework/prepare_full.py) prepares the release-source overlay, and [`build-framework.sh`](tools/build-framework.sh) builds it. |
| [`columnar/`](columnar/) and [`jet_transport.py`](jet_transport.py) | **Path C**. [`columnar/jets.py`](columnar/jets.py) reads PHYSLITE with uproot, calls CorrectionLib on batches of jets, and fills the histograms. The transport binds chunk data to the native provider. |
| [`provider/`](provider/) and [`portable/`](portable/) | Shared C++ corrections called by B and C. [`registry.cxx`](provider/registry.cxx) registers the native nodes. [`portable/uncertainty.cxx`](portable/uncertainty.cxx) contains the direct uncertainty evaluator. The portable code also contains ROOT-free numerical primitives and their tests. |
| [`exporters/`](exporters/) | Extracts the release calibration data and builds the CorrectionLib payload. Start with [`bundle.py`](exporters/bundle.py) for export, [`pack_eta_jes.py`](exporters/pack_eta_jes.py) for packing, and [`partition_payload.py`](exporters/partition_payload.py) for single-radius payloads. |
| [`packaging/`](packaging/) | Assembles the wheel and generates the standalone script. [`build_demo.py`](packaging/build_demo.py) is the packager, and [`build_jet_shipment.sh`](tools/build_jet_shipment.sh) builds and installs the shipment for testing. |
| [`tools/`](tools/) | Build, validation, timing, and profiling drivers. Start with [`qualify_shipment.py`](tools/qualify_shipment.py) for final acceptance or [`benchmark_campaign.py`](tools/benchmark_campaign.py) for the matched A/B/C timings. |
| [`provenance/`](provenance/) and [`source/correctionlib`](source/correctionlib) | Pinned source identities, evaluator and provider patch records with trial fixtures, and the CorrectionLib Git submodule. |
