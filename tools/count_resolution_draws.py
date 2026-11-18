"""Observe the instrumented original-stream cache in a complete columnar workflow."""
import ctypes
import hashlib
import json
from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from columnar import jets


COUNTERS = ("hits", "misses", "standard_draws", "guarded_restarts", "negative_restarts",
            "nonfinite_restarts", "scalar_seeds", "scalar_draws", "scalar_rejections")


def main(original: Path, histogram: Path, output: Path, provider: Path, payload: Path, provenance: Path,
         scope: jets.Scope = jets.Scope.both, events: int = 20000):
    if output.exists() or histogram.exists():
        raise FileExistsError("Supply new counter and histogram outputs")
    native = ctypes.CDLL(str(provider.resolve(strict=True)))
    counter = native.jets_resolution_rng_counts
    counter.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t)
    counter.restype = ctypes.c_size_t
    if counter(None, 0) != len(COUNTERS):
        raise ValueError("Counter schema differs")
    data = (ctypes.c_uint64 * len(COUNTERS))()
    counter(data, len(COUNTERS))
    if any(data):
        raise ValueError("Instrumented provider is already in use")
    jets.main(original, histogram, provider=provider, payload=payload, provenance=provenance,
              scope=scope, mode=jets.Mode.full, events=events)
    counter(data, len(COUNTERS))
    values = dict(zip(COUNTERS, data, strict=True))
    if values["standard_draws"] != values["misses"] or values["scalar_draws"] != values["scalar_seeds"] + values["scalar_rejections"]:
        raise ValueError("Inconsistent RNG accounting")
    with provider.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with output.open("x") as stream:
        json.dump({"scope": scope.value, "events": events, "provider_sha256": digest, "cache_bytes_per_thread": 65536 * 16,
                   "process": jets.check_execution(), "counts": values}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    typer.run(main)
