"""Select collection payloads without reserializing any correction node."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import typer

APP = typer.Typer(add_completion=False)
DECODER = json.JSONDecoder()
GROUPS = {
    "shared": frozenset(("jet_event_inputs", "jet_pileup_truth", "jet_event_sf")),
    "small": frozenset((
        "small_pileup", "small_uncertainty", "small_gsc", "small_eta_jes",
        "small_nnjvt_update", "small_nnjvt_select", "small_nnjvt_sf",
    )),
    "large": frozenset((
        "large_selection", "large_uncertainty", "large_eta_jes", "large_jms", "large_jmr",
    )),
}
SCOPES = {
    "small": GROUPS["shared"] | GROUPS["small"],
    "large": GROUPS["shared"] | GROUPS["large"],
}


@dataclass(frozen=True)
class Span:
    name: str
    first: int
    stop: int


@dataclass(frozen=True)
class Corrections:
    prefix: str
    suffix: str
    items: list[Span]


def nonspace(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def delimiter(text: str, position: int, token: str) -> int:
    position = nonspace(text, position)
    if not text.startswith(token, position):
        raise ValueError(f"Expected {token!r} at character {position}")
    return position + 1


def correction_spans(text: str, position: int) -> tuple[list[Span], int, int]:
    array = delimiter(text, position, "[")
    cursor = nonspace(text, array)
    items = []
    while not text.startswith("]", cursor):
        first = cursor
        item, cursor = DECODER.raw_decode(text, cursor)
        if not isinstance(item, dict):
            raise ValueError(f"Expected a correction object at character {first}")
        name = item.get("name")
        data = item.get("data")
        if (not isinstance(name, str) or not name or not isinstance(data, dict)
                or data.get("nodetype") != "extension"):
            raise ValueError(f"Expected a named jet extension correction: {name!r}")
        items.append(Span(name, first, cursor))
        del item, data
        cursor = nonspace(text, cursor)
        if text.startswith("]", cursor):
            break
        cursor = nonspace(text, delimiter(text, cursor, ","))
        if text.startswith("]", cursor):
            raise ValueError("Trailing comma in canonical corrections")
    return items, array, delimiter(text, cursor, "]")


def scan(text: str) -> Corrections:
    cursor = nonspace(text, delimiter(text, 0, "{"))
    keys = set()
    entries = None
    array = after = 0
    schema = None
    while not text.startswith("}", cursor):
        key, cursor = DECODER.raw_decode(text, cursor)
        if not isinstance(key, str) or key in keys:
            raise ValueError(f"Unexpected canonical key {key!r}")
        keys.add(key)
        cursor = nonspace(text, delimiter(text, cursor, ":"))
        if key == "corrections":
            entries, array, after = correction_spans(text, cursor)
            cursor = after
        else:
            value, cursor = DECODER.raw_decode(text, cursor)
            if key == "schema_version":
                schema = value
            elif key == "compound_corrections" and value:
                raise ValueError("This jet workload partition has no compound-correction dependency plan")
            elif key not in ("description", "compound_corrections"):
                raise ValueError(f"Unexpected canonical key {key!r}")
        cursor = nonspace(text, cursor)
        if text.startswith("}", cursor):
            break
        cursor = nonspace(text, delimiter(text, cursor, ","))
        if text.startswith("}", cursor):
            raise ValueError("Trailing comma in canonical CorrectionSet")
    if nonspace(text, delimiter(text, cursor, "}")) != len(text) or schema != 3 or entries is None:
        raise ValueError("Expected one complete schema-v3 jet CorrectionSet")
    if len({item.name for item in entries}) != len(entries):
        raise ValueError("Duplicate correction names in canonical jet payload")
    return Corrections(text[:array], text[after - 1:], entries)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    with path.open("rb") as stream:
        # hashlib.file_digest isn't available in the guest's stock Python 3.9.
        accumulator = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            accumulator.update(block)
        return accumulator.hexdigest()


def write_selection(output: Path, source: str, scan: Corrections, names: frozenset[str]) -> dict:
    selected = [item for item in scan.items if item.name in names]
    with output.open("xb") as stream:
        stream.write(scan.prefix.encode())
        for index, item in enumerate(selected):
            if index:
                stream.write(b",")
            stream.write(source[item.first:item.stop].encode())
        stream.write(scan.suffix.encode())
    return {"bytes": output.stat().st_size, "sha256": sha_file(output),
            "corrections": [item.name for item in selected]}


def relative_link(source: Path, output: Path) -> None:
    output.symlink_to(os.path.relpath(source, output.parent))


@APP.command()
def export(canonical: Path, result: Path) -> None:
    """Write disjoint tables and opt-in small/large runtime inputs in a new result directory."""
    canonical = canonical.resolve(strict=True)
    provenance = canonical.with_name("provenance.json").resolve(strict=True)
    data = canonical.read_bytes()
    source = data.decode("utf-8")
    entries = scan(source)
    reference_names = frozenset(item.name for item in entries.items)
    expected = frozenset.union(*GROUPS.values())
    if len(expected) != sum(map(len, GROUPS.values())) or reference_names != expected:
        raise ValueError(f"Unexpected canonical correction inventory. Missing {sorted(expected - reference_names)}, "
                         f"extra {sorted(reference_names - expected)}")

    result = result.resolve()
    result.mkdir()
    full = result / "canonical"
    full.mkdir()
    relative_link(canonical, full / "jets.json")
    relative_link(provenance, full / "provenance.json")
    fragments = result / "fragments"
    fragments.mkdir()
    artifact_manifest = {}
    for group, names in GROUPS.items():
        target = fragments / f"{group}.json"
        artifact_manifest[str(target.relative_to(result))] = write_selection(target, source, entries, names)
    for group, names in SCOPES.items():
        target = result / "scopes" / group
        target.mkdir(parents=True)
        relative_link(provenance, target / "provenance.json")
        path = target / "jets.json"
        artifact_manifest[str(path.relative_to(result))] = write_selection(path, source, entries, names)

    checks = {
        item.name: {
            "characters": [item.first, item.stop],
            "sha256": sha_bytes(source[item.first:item.stop].encode()),
            "bytes": len(source[item.first:item.stop].encode()),
        }
        for item in entries.items
    }
    for name, artifact in artifact_manifest.items():
        actual = (result / name).read_bytes().decode("utf-8")
        scan_result = scan(actual)
        retained = [item.name for item in scan_result.items]
        if retained != artifact["corrections"]:
            raise ValueError(f"Correction selection or order disagrees: {name}")
        for item in scan_result.items:
            if sha_bytes(actual[item.first:item.stop].encode()) != checks[item.name]["sha256"]:
                raise ValueError(f"Raw canonical bytes changed: {name} / {item.name}")

    manifest = {
        "canonical": {"path": str(canonical), "bytes": len(data), "sha256": sha_bytes(data)},
        "provenance": {"path": str(provenance), "sha256": sha_file(provenance)},
        "canonical_order": [item.name for item in entries.items],
        "nodes": checks,
        "artifacts": artifact_manifest,
        "raw_node_identity_verified": True,
        "native_load_paths": {
            "small": "scopes/small/jets.json", "large": "scopes/large/jets.json",
            "combined": "canonical/jets.json",
        },
        "package_storage": "Disjoint fragments OR canonical payload plus its scope index. Projections are development fixtures.",
    }
    (result / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for filename, properties in artifact_manifest.items():
        print(f"{filename}: {properties['bytes']} bytes, {len(properties['corrections'])} corrections")


if __name__ == "__main__":
    APP()
