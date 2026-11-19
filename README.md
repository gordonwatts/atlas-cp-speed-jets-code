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
