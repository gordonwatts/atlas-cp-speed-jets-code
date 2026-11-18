"""Inventory distinct source files and the exact shipped wheel members in bytes."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import typer


def source_receipts(value):
    if isinstance(value, dict):
        if {"sha256", "bytes", "resolved_path"} <= value.keys():
            yield value
        else:
            for entry in value.values():
                yield from source_receipts(entry)
    elif isinstance(value, list):
        for entry in value:
            yield from source_receipts(entry)


def main(provenance: Path, wheel: Path, output: Path):
    sources = json.loads(provenance.read_text())["sources"]
    files = {}
    for stage, record in sources.items():
        for receipt in source_receipts(record):
            path = Path(receipt["resolved_path"]).resolve(strict=True)
            item = files.setdefault(str(path), {
                "path": str(path), "logical_name": receipt.get("path_resolver_name", receipt.get("logical_path")),
                "sha256": receipt["sha256"], "bytes": receipt["bytes"], "stages": [],
            })
            if (path.stat().st_size, item["sha256"], item["bytes"]) != (
                    receipt["bytes"], receipt["sha256"], receipt["bytes"]):
                raise RuntimeError(f"Inconsistent source file receipt for {path}")
            if stage not in item["stages"]:
                item["stages"].append(stage)
    totals = Counter()
    for item in files.values():
        kind = {"root":"ROOT containers", "config":"tool settings", "json":"JSON model or cuts"}.get(
            Path(item["path"]).suffix[1:].lower(), "other settings")
        item["category"] = kind
        totals[kind] += item["bytes"]

    with ZipFile(wheel) as archive:
        members = {
            info.filename: {"uncompressed_bytes": info.file_size, "compressed_bytes": info.compress_size,
                            "zip_method": info.compress_type}
            for info in archive.infolist() if not info.is_dir()
        }
    with wheel.open("rb") as stream:
        wheel_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "source": {"method": "Distinct resolved files consumed by the included original configurations; ROOT byte counts include all keys in each required container.",
                   "files": sorted(files.values(), key=lambda record: record["path"]),
                   "file_count": len(files), "total_bytes": sum(totals.values()), "by_category_bytes": dict(totals)},
        "wheel": {"path": str(wheel.resolve()), "bytes": wheel.stat().st_size, "sha256": wheel_sha,
                  "members": members, "method": "ZIP member sizes exclude archive directory/header overhead."},
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    typer.run(main)
