"""Emit the exact variable-axis geometry from the archived complete payload."""
import json
from pathlib import Path
import sys


def histograms(node, location):
    if isinstance(node, dict):
        if {"axes", "class", "content"} <= node.keys():
            yield location, node["axes"]
        else:
            for key, child in node.items():
                yield from histograms(child, f"{location}/{key}")
    elif isinstance(node, list):
        for index, child in enumerate(node):
            yield from histograms(child, f"{location}/{index}")


def main(payload):
    document = json.loads(Path(payload).read_text())
    seen = set()
    for correction in document["corrections"]:
        for name, axes in histograms(correction["data"], correction["name"]):
            for dimension, axis in enumerate(axes):
                if not axis["edges"]:
                    continue
                values = tuple(float(value).hex() for value in (axis["min"], axis["max"], *axis["edges"]))
                if values in seen:
                    continue
                seen.add(values)
                print(f"{name} axis {dimension}")
                print(" ".join(values))


if __name__ == "__main__":
    main(sys.argv[1])
