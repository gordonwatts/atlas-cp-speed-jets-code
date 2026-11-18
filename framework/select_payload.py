"""Select a compatible compact CorrectionSet for the existing JETS_PAYLOAD hook."""

from enum import Enum
import json
from pathlib import Path
import sys

import typer

checkout = Path(__file__).resolve().parent.parent
if str(checkout) not in sys.path:
    sys.path.insert(0, str(checkout))
from exporters.payload_ranges import identify, materialize

APP = typer.Typer(add_completion=False)


class Scope(str, Enum):
    small = "small"
    large = "large"
    both = "both"


@APP.command()
def select(partition: Path, canonical: Path, scope: Scope) -> None:
    """Print a checked absolute path to an existing scope-matched artifact."""
    manifest = json.loads((partition / "manifest.json").read_text())
    actual = identify(canonical)
    if any(manifest["canonical"][field] != value for field, value in actual.items()):
        raise ValueError("The partition must be generated from this backend's canonical payload")
    if scope == Scope.both:
        print(canonical.resolve(strict=True))
        return

    relative = manifest["native_load_paths"][scope.value]
    target = (partition / relative).resolve(strict=True)
    recorded = manifest["artifacts"][relative]
    if any(recorded[field] != value for field, value in identify(target).items()):
        raise ValueError("Selected scoped artifact differs from the partition manifest")
    print(target)


@APP.command()
def project(canonical: Path, index: Path, scope: Scope, output: Path) -> None:
    """Write a fresh compact file with the same projector used by the visible C script."""
    layout = json.loads(index.read_text())
    if identify(canonical) != layout["canonical"]:
        raise ValueError("Scope index and backend payload identity disagree")
    materialize(canonical, index, scope.value, output)
    expected = layout["scopes"][scope.value]
    if any(expected[field] != value for field, value in identify(output).items()):
        raise ValueError("Projected CorrectionSet differs from the index")
    print(output.resolve(strict=True))


if __name__ == "__main__":
    APP()
