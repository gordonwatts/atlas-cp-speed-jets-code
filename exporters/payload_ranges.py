"""Reconstruct a compact CorrectionSet by copying selected canonical JSON spans."""

import hashlib
import json
from pathlib import Path

SCOPES = ("small", "large")
BLOCK = 1024 * 1024


def identify(path: Path) -> dict:
    with path.open("rb") as stream:
        return {"bytes": path.stat().st_size, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def make_index(canonical: Path, partition_manifest: Path, output: Path) -> None:
    """Convert verified character spans to byte offsets in the exact full artifact."""
    manifest = json.loads(partition_manifest.read_text())
    raw = canonical.read_bytes()
    source = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    if any(manifest["canonical"].get(key) != value for key, value in source.items()):
        raise ValueError("Scope manifest does not describe the canonical input")

    spans = {name: tuple(info["characters"]) for name, info in manifest["nodes"].items()}
    points = {point for span in spans.values() for point in span}
    ascii_only = raw.isascii()
    text = raw if ascii_only else raw.decode("utf-8")
    prefix = min(points)
    suffix = max(points)
    while text[prefix - 1:prefix].isspace():
        prefix -= 1
    while text[suffix:suffix + 1].isspace():
        suffix += 1
    offsets = sorted(points | {prefix, suffix})
    if ascii_only:
        byte_positions = {point: point for point in offsets}
    else:
        position = byte_position = 0
        byte_positions = {}
        for point in offsets:
            byte_position += len(text[position:point].encode())
            byte_positions[point] = byte_position
            position = point

    columns = {name: [byte_positions[first], byte_positions[stop]]
               for name, (first, stop) in spans.items()}
    for name, (start, stop) in columns.items():
        if hashlib.sha256(raw[start:stop]).hexdigest() != manifest["nodes"][name]["sha256"]:
            raise ValueError(f"Canonical byte offsets differ for {name}")

    scopes = {}
    for scope in SCOPES:
        path = manifest["native_load_paths"][scope]
        info = manifest["artifacts"][path]
        scopes[scope] = {
            "spans": [columns[name] for name in info["corrections"]],
            "bytes": info["bytes"],
            "sha256": info["sha256"],
        }
    specification = {
        "format": 1,
        "canonical": source,
        "envelope": [[0, byte_positions[prefix]], [byte_positions[suffix], len(raw)]],
        "scopes": scopes,
    }
    with output.open("x") as stream:
        json.dump(specification, stream, separators=(",", ":"))
        stream.write("\n")


def copy_range(source, emit, first: int, stop: int) -> None:
    source.seek(first)
    remaining = stop - first
    while remaining:
        data = source.read(min(BLOCK, remaining))
        if not data:
            raise EOFError("Canonical payload ended inside an indexed span")
        emit(data)
        remaining -= len(data)


def materialize(canonical: Path, index: Path, scope: str, output: Path) -> dict:
    """Copy the selected canonical bytes without parsing the unused correction nodes."""
    if scope not in SCOPES:
        raise ValueError(f"Compact payload applies to one collection, got {scope!r}")
    layout = json.loads(index.read_text())
    if layout.get("format") != 1:
        raise ValueError("Unsupported compact payload index")
    size = layout["canonical"]["bytes"]
    (prefix_first, prefix_stop), (suffix_first, suffix_stop) = layout["envelope"]
    subset = layout["scopes"][scope]
    spans = subset["spans"]
    if (not spans or prefix_first != 0 or suffix_stop != size
            or not 0 < prefix_stop <= suffix_first < size):
        raise ValueError("Invalid compact payload envelope")
    cursor = prefix_stop
    reconstructed_size = prefix_stop + size - suffix_first + len(spans) - 1
    for first, stop in spans:
        if not cursor <= first < stop <= suffix_first:
            raise ValueError("Invalid or overlapping compact payload spans")
        reconstructed_size += stop - first
        cursor = stop
    if reconstructed_size != subset["bytes"]:
        raise ValueError("Indexed compact payload dimensions disagree")

    digest = hashlib.sha256()
    with canonical.open("rb") as source:
        if source.seek(0, 2) != size:
            raise ValueError("The canonical payload size differs from the index")
        with output.open("xb") as destination:
            def emit(data: bytes) -> None:
                if destination.write(data) != len(data):
                    raise OSError("Incomplete write to compact CorrectionSet")
                digest.update(data)

            copy_range(source, emit, prefix_first, prefix_stop)
            for row, (first, stop) in enumerate(spans):
                if row:
                    emit(b",")
                copy_range(source, emit, first, stop)
            copy_range(source, emit, suffix_first, suffix_stop)
            byte_count = destination.tell()
    identity = {"bytes": byte_count, "sha256": digest.hexdigest()}
    if any(subset[field] != value for field, value in identity.items()):
        output.unlink()
        raise ValueError("Written compact CorrectionSet differs from its verified byte extent or digest")
    return identity
