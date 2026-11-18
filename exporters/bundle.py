"""Assemble the independently exported jet prescriptions in the AnalysisBase environment."""

import hashlib
import json
from pathlib import Path

from exporters.eta_jes import export_eta_jes
from exporters.large_jmr import export_large_jmr
from exporters.large_jms import export_large_jms
from exporters.small_jvt import export_small_jvt
from exporters.small_pileup import export_small_pileup
from exporters.small_gsc import export_small_gsc
from exporters.event_inputs import export_event_inputs
from exporters.jet_uncertainty import export_jet_uncertainties


def export_bundle(directory: Path):
    directory.mkdir(parents=False, exist_ok=False)
    corrections, sources = [], {}
    for name, export in [
        ("small_pileup", export_small_pileup),
        ("event", export_event_inputs),
        ("uncertainties", export_jet_uncertainties),
        ("small_gsc", export_small_gsc),
        ("small_eta_jes", lambda: export_eta_jes("small")),
        ("large_eta_jes", lambda: export_eta_jes("large")),
        ("large_jms", export_large_jms),
        ("large_jmr", export_large_jmr),
        ("small_nnjvt", export_small_jvt),
    ]:
        fragment, sources[name] = export()
        corrections.extend(fragment if isinstance(fragment, list) else [fragment])
    payload = {"schema_version": 3, "corrections": corrections}
    path = directory / "jets.json"
    path.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n")
    manifest = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "correction_names": [c["name"] for c in corrections], "sources": sources}
    (directory / "provenance.json").write_text(json.dumps(manifest, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
    import typer
    typer.run(export_bundle)
