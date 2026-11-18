"""Check raw payload projections with the installed native evaluator without processing events."""

import json
from pathlib import Path
import sys

from atlas_jet_demo.runtime import PROVIDER, audit_runtime, prepare_runtime

prepare_runtime()

import typer

APP = typer.Typer(add_completion=False)


def signatures(correction_set):
    return {
        key: {
            "inputs": [(variable.name, variable.type) for variable in correction_set[key].inputs],
            "output": (correction_set[key].output.name, correction_set[key].output.type),
            "version": correction_set[key].version,
        }
        for key in sorted(correction_set)
    }


@APP.command()
def run(partitions: Path, output: Path) -> None:
    """Load each opt-in set and check its available public signatures and exact scope."""
    from correctionlib import CorrectionSet, plugins

    manifest = json.loads((partitions / "manifest.json").read_text())
    plugins.enable_discovery(False)
    plugins.load_library(str(PROVIDER.resolve(strict=True)))
    canonical = signatures(CorrectionSet.from_file(str(partitions / "canonical/jets.json")))
    if set(canonical) != set(manifest["canonical_order"]):
        raise ValueError("Canonical inventory changed at native load")
    loaded = {"combined": canonical}
    for scope in ("small", "large"):
        relative = manifest["native_load_paths"][scope]
        expected_names = manifest["artifacts"][relative]["corrections"]
        compact = signatures(CorrectionSet.from_file(str(partitions / relative)))
        matched = {name: canonical[name] for name in expected_names}
        if compact != matched:
            raise ValueError(f"Native {scope} inventory or signatures differ")
        loaded[scope] = compact

    with output.open("x") as stream:
        json.dump({
            "passed": True,
            "python": sys.version,
            "scope_public_signatures": loaded,
            "native_libraries": audit_runtime(),
            "numerical_events_evaluated": 0,
        }, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    APP()
