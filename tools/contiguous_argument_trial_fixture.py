"""Exercise the argument-copy gate before timing the visible installed 006 CLI."""
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace

visible, expected = sys.argv[1:]
sys.argv[0] = visible  # Reproduce the exact installed CLI's private gate.
active = expected == "true"

from atlas_jet_demo.runtime import audit_runtime, prepare_runtime

prepare_runtime()
import numpy as np
from correctionlib import CorrectionSet, _core, highlevel, plugins
from atlas_jet_demo.runtime import PROVIDER

plugins.enable_discovery(False)
plugins.load_library(str(PROVIDER))
content = {"schema_version": 2, "corrections": [{"name": "test", "version": 1, "inputs": [
    {"name": "x", "type": "real"}, {"name": "y", "type": "real"}
], "output": {"name": "value", "type": "real"},
   "data": {"nodetype": "formula", "expression": "x + 2*y", "parser": "TFormula", "variables": ["x", "y"]}}]}
correction = CorrectionSet.from_string(json.dumps(content))["test"]


class Subclass(np.ndarray):
    pass


owned = np.array([0.0, -0.0, 1.5, 42., float("inf"), float("nan")])
readonly = owned.copy()
readonly.flags.writeable = False
other = np.arange(owned.size, dtype=np.float64)
cases = [
    ("owned", owned, other, active),
    ("readonly-owned", readonly, other, active),
    ("empty", np.empty(0), np.empty(0), active),
    ("singleton", np.ones(1), np.ones(1), active),
    ("native-float32", owned.astype(np.float32), other, active),
    ("owned-native-integer", other.astype(np.int64), other, active),
    ("nonowning", owned[:], other, False),
    ("strided", owned[::2], other[::2], False),
    ("negative-stride", owned[::-1], other, False),
    ("zero-stride", np.broadcast_to(np.array([2.]), owned.shape), other, False),
    ("subclass", owned.view(Subclass), other, False),
    ("non-native-endian", owned.astype(">f8"), other, False),
    ("broadcast", owned[:1], other, False),
    ("zero-dimensional", np.array(-0.0), other, False),
    ("multidimensional", owned.reshape(3, 2), other.reshape(3, 2), False),
]
result_hash = sha256()
for name, array, argument, expect_share in cases:
    forwarded = []

    def capture(x, y):
        forwarded.append(x)
        return x.copy()

    highlevel._evaluate(SimpleNamespace(_base=SimpleNamespace(evalv=capture)), array, argument)
    empty_borrow = not array.size and forwarded[0] is array
    if (empty_borrow or np.shares_memory(array, forwarded[0])) != expect_share:
        raise AssertionError(f"Wrong copy policy for {name}")
    before = array.tobytes()
    value = correction.evaluate(array, argument)
    if array.tobytes() != before:
        raise AssertionError(f"Input was mutated for {name}")
    result_hash.update(str(value.shape).encode())
    result_hash.update(value.tobytes())

# A stand-in for a mutation while evalv is active. The supported general API
# must pass a snapshot to evalv, even if a caller can still modify its array.
if not active:
    sample = np.array([3.])
    original = sample.copy()

    def mutate(x, y):
        sample[0] = -5.
        if x[0] != original[0]:
            raise AssertionError("General API input is no longer a copied snapshot")
        return x

    highlevel._evaluate(SimpleNamespace(_base=SimpleNamespace(evalv=mutate)), sample, 1.)

print(json.dumps({"passed": True, "output_sha256": result_hash.hexdigest(), "checked_cases": len(cases),
                  "highlevel": highlevel.__file__, "binding_real_path": str(Path(_core.__file__).resolve()),
                  "runtime_maps": audit_runtime()}))
