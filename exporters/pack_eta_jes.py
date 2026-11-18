"""Create an opt-in packed EtaJES CorrectionSet from the canonical decimal JSON."""

from base64 import b64decode, b64encode
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import struct
from typing import Annotated

import typer

from exporters.partition_payload import scan, sha_file

ENCODING = "base64-ieee754-binary32-le-v1"
TARGETS = frozenset(("small_eta_jes", "large_eta_jes"))
CONTENT = re.compile(r'"content"\s*:\s*(\[)')
FLOAT = struct.Struct("<f")


def unique_object(items):
    obj = dict(items)
    if len(obj) != len(items):
        raise ValueError("Duplicate JSON object member in EtaJES correction")
    return obj


def reject_constant(value):
    raise ValueError(f"Nonstandard JSON number in EtaJES correction: {value}")


DECODER = json.JSONDecoder(object_pairs_hook=unique_object, parse_constant=reject_constant)


@dataclass(frozen=True)
class Patch:
    first: int
    stop: int
    replacement: bytes
    proof: dict


def pack_cells(original):
    """Encode exact finite binary32 words and verify their original decimal values."""
    decoded = bytearray(len(original) * FLOAT.size)
    for i, value in enumerate(original):
        if type(value) not in (int, float):
            raise ValueError(f"Expected a numeric TH1F cell at index {i}")
        try:
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite TH1F cell at index {i}")
            FLOAT.pack_into(decoded, FLOAT.size * i, value)
        except (OverflowError, struct.error) as error:
            raise ValueError(f"TH1F cell at index {i} is outside binary32") from error

    data = b64encode(decoded)
    recovered = b64decode(data, validate=True)
    if recovered != decoded:
        raise ValueError("Standard base64 failed to recover the original binary32 words")

    negative_zero = 0
    for i, (value, (restored,)) in enumerate(zip(original, FLOAT.iter_unpack(recovered), strict=True)):
        if restored != value or (value == 0 and math.copysign(1.0, restored) != math.copysign(1.0, value)):
            raise ValueError(f"TH1F decimal cell at index {i} is not an exact stored binary32 value")
        negative_zero += restored == 0 and math.copysign(1.0, restored) < 0
    packed = {"encoding": ENCODING, "count": len(original), "data": data.decode("ascii")}
    return packed, {
        "cells": len(original), "raw_bytes": len(recovered), "base64_bytes": len(data),
        "binary32_sha256": sha256(recovered).hexdigest(), "negative_zero_cells": negative_zero,
    }


def content_patches(source, entry, names):
    config_root, stop = DECODER.raw_decode(source, entry.first)
    node = config_root["data"]
    if (stop != entry.stop or config_root["name"] != entry.name or node["nodetype"] != "extension"
            or node["type"] != "org.atlas.jets.eta_jes" or node["version"] != 1):
        raise ValueError(f"Unsupported EtaJES correction shape at {entry.name}")
    cfg = node["config"]
    tables = cfg["bins"]
    members = list(CONTENT.finditer(source, entry.first, entry.stop))
    if len(tables) != len(names) or len(cfg["eta_edges"]) != len(tables) + 1 or len(members) != len(tables):
        raise ValueError(f"Unexpected EtaJES cardinality in {entry.name}")

    patches, preserved = [], []
    for index, (table, member, name) in enumerate(zip(tables, members, names, strict=True)):
        histogram = table["response"]
        axes = histogram["axes"]
        content = histogram["content"]
        if (set(table) != {"response", "eta_coefficients", "freeze_at"}
                or set(histogram) != {"class", "axes", "content", "sumw2"}
                or len(axes) != 1 or type(axes[0]["nbins"]) is not int
                or axes[0]["nbins"] < 1 or not isinstance(content, list)
                or len(content) != axes[0]["nbins"] + 2):
            raise ValueError(f"Malformed EtaJES response {entry.name}/{index}")
        first = member.start(1)
        cells, stop = DECODER.raw_decode(source, first)
        if stop > entry.stop or cells != content:
            raise ValueError(f"Response content span does not address {entry.name}/{index}")
        if histogram["class"] != "TH1F":
            preserved.append({"index": index, "name": name, "class": histogram["class"]})
            continue
        try:
            packed, proof = pack_cells(cells)
        except ValueError as error:
            raise ValueError(f"{entry.name}/{index} {name}: {error}") from error
        encoded = json.dumps(packed, allow_nan=False, separators=(",", ":")).encode("ascii")
        proof.update({
            "correction": entry.name, "index": index, "name": name, "class": "TH1F",
            "nbins": axes[0]["nbins"], "includes_underflow_and_overflow": True,
            "canonical_content_bytes": stop - first, "packed_content_bytes": len(encoded),
        })
        patches.append(Patch(first, stop, encoded, proof))
    if not patches:
        raise ValueError(f"No eligible TH1F response in {entry.name}")
    return patches, preserved


def write_payload(output, source, patches):
    """Copy every source span outside the exact content values as its original UTF-8 bytes."""
    identity, untouched = sha256(), sha256()
    cursor, canonical_byte, output_byte = 0, 0, 0
    with output.open("xb") as stream:
        for patch in patches:
            if not cursor <= patch.first < patch.stop <= len(source):
                raise ValueError("Packed histogram edits overlap or leave the canonical document")
            unchanged = source[cursor:patch.first].encode("utf-8")
            original = source[patch.first:patch.stop].encode("utf-8")
            canonical_byte += len(unchanged)
            output_byte += len(unchanged)
            patch.proof["canonical_content_byte_span"] = [canonical_byte, canonical_byte + len(original)]
            patch.proof["packed_content_byte_span"] = [output_byte, output_byte + len(patch.replacement)]
            canonical_byte += len(original)
            output_byte += len(patch.replacement)
            stream.write(unchanged)
            stream.write(patch.replacement)
            untouched.update(unchanged)
            identity.update(unchanged)
            identity.update(patch.replacement)
            cursor = patch.stop
        suffix = source[cursor:].encode("utf-8")
        stream.write(suffix)
        identity.update(suffix)
        untouched.update(suffix)
    packed = {"path": str(output.resolve()), "bytes": output.stat().st_size,
              "sha256": identity.hexdigest()}
    return packed, untouched.hexdigest()


def main(
    canonical: Path,
    result: Path,
    provenance: Annotated[Path | None, typer.Option(help="Canonical payload source provenance")] = None,
):
    """Pack only TH1F EtaJES response cells and retain their exact binary32 proof."""
    if result.exists():
        raise FileExistsError(result)
    canonical = canonical.resolve(strict=True)
    provenance = provenance or canonical.with_name("provenance.json")
    inherited_provenance = json.loads(provenance.read_text(encoding="utf-8"))
    raw = canonical.read_bytes()
    canonical_id = {"path": str(canonical), "bytes": len(raw), "sha256": sha256(raw).hexdigest()}
    source = raw.decode("utf-8")
    del raw
    entries = scan(source).items
    if TARGETS - {entry.name for entry in entries}:
        raise ValueError("Expected both canonical EtaJES corrections")
    patches, preserved = [], {}
    for entry in entries:
        if entry.name in TARGETS:
            names = inherited_provenance["sources"][entry.name]["ordered_histograms"]
            edits, other_types = content_patches(source, entry, names)
            patches.extend(edits)
            preserved[entry.name] = other_types

    result.mkdir()
    packed, unchanged_digest = write_payload(result / "jets.json", source, patches)
    inherited_sha = inherited_provenance.get("sha256")
    inherited_provenance.update({
        "sha256": packed["sha256"],
        "packed_eta_jes": {
            "encoding": ENCODING, "canonical_payload": canonical_id, "packing_receipt": "packing.json",
        },
    })
    with (result / "provenance.json").open("x", encoding="utf-8") as output:
        json.dump(inherited_provenance, output, indent=2, allow_nan=False)
        output.write("\n")
    receipt = {
        "encoding": ENCODING, "canonical": canonical_id, "packed": packed,
        "canonical_provenance": {"path": str(provenance.resolve()), "sha256": sha_file(provenance),
                                 "inherited_payload_sha256": inherited_sha},
        "common_utf8_segments_sha256": unchanged_digest,
        "untouched": "All original bytes outside the listed TH1F content values",
        "geometry_coefficients_sumw2": "Retained in the original UTF-8 bytes",
        "response_cells": sum(patch.proof["cells"] for patch in patches),
        "preserved_non_TH1F": preserved, "tables": [patch.proof for patch in patches],
    }
    with (result / "packing.json").open("x", encoding="utf-8") as output:
        json.dump(receipt, output, indent=2, allow_nan=False)
        output.write("\n")
    typer.echo(f"Packed {len(patches)} tables, {receipt['response_cells']} exact binary32 cells.")
    typer.echo(f"CorrectionSet sizes {canonical_id['bytes']} -> {packed['bytes']} bytes")


if __name__ == "__main__":
    typer.run(main)
