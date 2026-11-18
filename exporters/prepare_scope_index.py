"""Create the compact runtime index and verify all reconstructed bytes independently."""

import json
from pathlib import Path
import tempfile

import typer

from payload_ranges import SCOPES, identify, make_index, materialize

APP = typer.Typer(add_completion=False)


@APP.command()
def build(canonical: Path, partition: Path, output: Path) -> None:
    """Index one canonical file and verify both scoped documents without evaluating it."""
    make_index(canonical, partition / "manifest.json", output)
    index = json.loads(output.read_text())
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        for scope in SCOPES:
            materialized = Path(scratch) / f"{scope}.json"
            materialize(canonical, output, scope, materialized)
            got = identify(materialized)
            expected = {name: index["scopes"][scope][name] for name in ("bytes", "sha256")}
            if got != expected:
                raise ValueError(f"Reconstructed {scope} correction bytes differ from the prebuilt set")
            print(f"{scope}: {got['bytes']} bytes, SHA-256 {got['sha256']}")
    print(f"Index: {output.stat().st_size} bytes")


if __name__ == "__main__":
    APP()
