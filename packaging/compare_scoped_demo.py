"""Check and measure opt-in indexed startup with one pristine user-facing wheel."""

from enum import Enum
import json
import os
from pathlib import Path
import sys
from typing import Annotated

import typer

from check_demo import CLEAN_ENV, digest, run

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/reference"))
from benchmark import check_quality, check_pair, run_leg

APP = typer.Typer(add_completion=False)
BRANCH_DIGEST = "Input branch SHA-256:"
PROJECTION_RECEIPTS = {"sha256": "Payload materialized SHA-256: ", "bytes": "Payload materialized bytes: "}


class Scope(str, Enum):
    small = "small"
    large = "large"
    both = "both"


def require_cpu():
    if os.sched_getaffinity(0) != {3} or len(list(Path("/proc/self/task").iterdir())) != 1:
        raise ValueError("Use reserved CPU3 and one worker thread")


def installed_commands(installation: Path):
    record = json.loads((installation / "installations.json").read_text())
    for front_end in ("pip", "uv"):
        for member in ("wheel", "script"):
            if digest(installation / front_end / record[member]) != record[f"{member}_sha256"]:
                raise ValueError(f"Candidate {front_end} {member} differs from the transferred identity")
    return {
        "pip": [installation / "pip/env/bin/python", installation / "pip" / record["script"]],
        "uv": [record["uv"], "run", "--no-project", "--no-config", "--native-tls",
               "--no-python-downloads", "--python", record["python"], "--cache-dir",
               installation / "uv/cache", installation / "uv" / record["script"]],
    }, record


def branch_digest(log: Path) -> str:
    value, = (line.split()[-1] for line in log.read_text().splitlines() if line.startswith(BRANCH_DIGEST))
    return value


def installed_index(installation: Path) -> dict:
    path, = (installation / "pip/env/lib").glob("python*/site-packages/atlas_jet_demo/data/payload-index.json")
    return json.loads(path.read_text())["scopes"]


def projection_receipt(log: Path, expected: dict | None) -> dict:
    lines = log.read_text().splitlines()
    found = {key: [line.removeprefix(prefix) for line in lines if line.startswith(prefix)]
             for key, prefix in PROJECTION_RECEIPTS.items()}
    if expected is None:
        if any(found.values()):
            raise ValueError(f"Unexpected projected payload in {log}")
        return {}
    if any(found[key] != [str(expected[key])] for key in found):
        raise ValueError(f"Projection byte identity differs from the checked index in {log}")
    return {key: expected[key] for key in found}


def compare(expected: Path, actual: Path) -> None:
    import numpy as np

    with np.load(expected, allow_pickle=False) as left, np.load(actual, allow_pickle=False) as right:
        if set(left.files) != set(right.files):
            raise ValueError(f"Different histogram outputs: {expected} and {actual}")
        for field in left.files:
            np.testing.assert_array_equal(left[field], right[field], err_msg=field)


def command(args, original: Path, destination: Path, scope: Scope, mode: str, events: int, options=()):
    return [*map(str, args), str(original), str(destination), "--scope", scope.value,
            "--mode", mode, "--events", str(events), "--matched-read", *options]


def record(path: Path, data) -> None:
    with path.open("x") as out:
        json.dump(data, out, indent=2, allow_nan=False)
        out.write("\n")


@APP.command()
def check(
    installation: Path,
    original: Path,
    authority: Path,
    scope: Scope,
    output: Path,
    events: Annotated[int, typer.Option(min=1)] = 32,
    verify_projection_identity: bool = False,
) -> None:
    """Verify both pristine front ends through the packaged compact CLI and stock histograms."""
    require_cpu()
    installation, original, authority, output = (path.resolve() for path in (installation, original, authority, output))
    commands, frozen = installed_commands(installation)
    output.mkdir()
    defaults = frozen.get("demo_defaults")
    canonical_options = ["--no-compact-payload", "--no-reuse-nominal-nnjvt-update"] if defaults else []
    cases = [
        ("pip", "canonical-full", "full", canonical_options, False),
        ("pip", "indexed-full", "full", ["--compact-payload"], True),
        ("pip", "indexed-nominal", "nominal", ["--compact-payload"], True),
        ("uv", "indexed-full", "full", ["--compact-payload"], True),
    ]
    if defaults:
        cases.append(("pip", "selected-default-full", "full", [],
                      scope.value in defaults["compact_payload_default_scopes"]))
    digests = {}
    projections = {}
    if verify_projection_identity:
        expected_scope = installed_index(installation).get(scope.value)
    for executable, stage, mode, options, expect_projection in cases:
        name = f"{executable}-{stage}"
        path = output / f"{name}.npz"
        run(command(commands[executable], original, path, scope, mode, events, options), output, name, 3,
            {"ATLAS_JET_LIBRARY_AUDIT": str(output / f"{name}-library-maps.json")})
        log = output / f"{name}.log"
        digests[name] = branch_digest(log)
        if verify_projection_identity:
            projections[name] = projection_receipt(log, expected_scope if expect_projection else None)
    authority_full = output / "pip-canonical-full.npz"
    compare(authority_full, output / "pip-indexed-full.npz")
    compare(authority_full, output / "uv-indexed-full.npz")
    compare(authority, authority_full)
    if defaults:
        compare(authority_full, output / "pip-selected-default-full.npz")
    check_pair(output / "pip-indexed-nominal.npz", authority_full)
    if len(set(digests.values())) != 1:
        raise ValueError(f"Resolved physical branch sets changed: {digests}")
    record(output / "comparison.json", {
        "installation": frozen,
        "scope": scope.value,
        "events": events,
        "original": str(original),
        "stock_full_histogram": {"path": str(authority), "sha256": digest(authority)},
        "branch_digests": digests,
        "projection_identities": projections,
        "all_histogram_fields_exact": True,
        "matched_nominal_nosys_exact": True,
    })
    require_cpu()


@APP.command()
def measure(
    installation: Path,
    original: Path,
    partition: Path,
    scope: Scope,
    output: Path,
    events: Annotated[int, typer.Option(min=0)] = 0,
    rounds: Annotated[int, typer.Option(min=4)] = 4,
    verify_projection_identity: bool = False,
) -> None:
    """Measure fresh canonical, prebuilt and reconstructed document startup through one script."""
    require_cpu()
    if scope == Scope.both:
        raise ValueError("The packaged combined workload uses the canonical input directly")
    installation, original, partition, output = (path.resolve() for path in (installation, original, partition, output))
    commands, frozen = installed_commands(installation)
    manifest = json.loads((partition / "manifest.json").read_text())
    relative = manifest["native_load_paths"][scope.value]
    prebuilt = partition / relative
    if digest(prebuilt) != manifest["artifacts"][relative]["sha256"]:
        raise ValueError("Prebuilt payload differs from the checked scope partition")
    output.mkdir()
    options = {
        "canonical": ["--no-compact-payload"] if "demo_defaults" in frozen else [],
        "prebuilt": ["--payload", str(prebuilt)],
        "indexed": ["--compact-payload"],
    }
    names = list(options)
    grouped = {variant: [] for variant in options}
    authority = None
    branches = None
    projections = {}
    if verify_projection_identity:
        expected_scope = installed_index(installation)[scope.value]
    for iteration in range(rounds):
        offset = iteration % len(options)
        for variant in names[offset:] + names[:offset]:
            work = output / f"{iteration:02d}-{variant}"
            work.mkdir()
            histogram = work / "histograms.npz"
            result = run_leg(command(commands["pip"], original, histogram, scope, "full", events, options[variant]),
                             CLEAN_ENV, work, "startup", event_source=original)
            log = work / "startup.log"
            selected = branch_digest(log)
            if verify_projection_identity:
                projections[work.name] = projection_receipt(log, expected_scope if variant == "indexed" else None)
            if authority is None:
                authority = histogram
                branches = selected
            elif selected != branches:
                raise ValueError("The load choice changed the selected input branches")
            compare(authority, histogram)
            grouped[variant].append(result)
    record(output / "summary.json", {
        "installation": frozen,
        "scope": scope.value,
        "events": events,
        "rounds": rounds,
        "branch_digest": branches,
        "projection_identities": projections,
        "all_histograms_exact": True,
        "prebuilt_payload": {"path": str(prebuilt), **manifest["artifacts"][relative]},
        "includes_private_temporary_file_construction": ["indexed"],
        "quality": {variant: check_quality(rows) for variant, rows in grouped.items()},
        "resource_review_required": True,
    })
    require_cpu()


if __name__ == "__main__":
    APP()
