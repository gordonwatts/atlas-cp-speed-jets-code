"""Exercise the opt-in signed-pair lease through the original CorrectionLib APIs."""

import ctypes as ct
import json
from pathlib import Path
import sys

from columnar.jets import check_execution, native_runtime
from correctionlib import CorrectionSet, plugins
from exporters.jet_uncertainty import correction_def
from jet_transport import JetBuffers
import numpy as np


def histogram(contents, dimensions=1):
    axes = [{"nbins": 2, "min": 0, "max": 2000, "edges": [0, 100, 2000], "range": None}]
    axes.extend({"nbins": 2, "min": -7, "max": 7, "edges": [], "range": None} for _ in range(dimensions - 1))
    return {"class": f"TH{dimensions}F", "axes": axes, "content": contents, "sumw2": []}


def model(value, invalid=False):
    def component(table, scale="fourvec", kind="table", validity=""):
        return {"kind": kind, "scale": scale, "param": "Pt", "mode": "None", "calo": False,
                "labels": [], "tables": [table], "validity": validity}

    def group(*components):
        return {"correlation": "Correlated", "components": components}

    groups = {
        "shared": group(component("shape")),
        "bflavour": group(component("shape", kind="bjes_response")),
        "resolution": group(component("resolution", "ptresabs")),
        "multiscale": group(component("shape"), component("resolution", "ptresabs")),
        "guard": group(component("shape", validity="mask")),
    }
    states = {}
    for name in groups:
        for stem, pseudo in (("", False), ("-pd", True)):
            # Deliberately arbitrary spelling and opposed parameter values.
            states[name + stem + "/one"] = {"basename": name, "parameter": -1, "pseudo_data": pseudo}
            states[name + stem + "/two"] = {"basename": name, "parameter": 1, "pseudo_data": pseudo}
    states["non-unit"] = {"basename": "shared", "parameter": 2, "pseudo_data": False}
    return {
        "mev_to_gev": 0.001, "rng_user_seed": 0, "mc_phi_seed_multiplier": 2e5,
        "pseudo_data_phi_seed_multiplier": 1.23e5, "mc_zero_seed": 45583453, "pseudo_data_zero_seed": 34545654,
        "abs_eta_gluon_fraction": False, "states": states, "analysis_histograms": {},
        "histograms": {
            "shape": histogram([0.0, value, value * 2, 0.0]),
            "resolution": histogram([0.0, 0.01, 0.02, 0.0]),
            "mask": histogram([0.0, 0.0, 1.0, 0.0]),
            "nominal": histogram([0.06] * 16, 2),
            "file_mask": histogram([0.0] * 64, 3),
        },
        "portable_plan": {
            "groups": groups, "file_validity": "file_mask" if invalid else "",
            "ref_mu": 0, "ref_npv": 0, "smear_only_mc": False,
            "resolutions": {"MC": {"table": "nominal", "mode": "OnlyX"},
                            "Data": {"table": "nominal", "mode": "OnlyX"}},
        },
    }


def vector(values):
    points = np.asarray(values, dtype=np.float64)
    return [
        points, 0.2, 0.7123457, 12000.0,
        np.zeros(len(points)), 23.0, np.array([1, 1, 1, 0]), np.array([1, 1, 0, 1]),
        np.array([5, 0, 5, 0]), np.array([1, 0, 1, 1]), 0, 0,
        0.0, 0.0, 0.0, 0.0, 0,
    ]


def equal(observed, expected):
    np.testing.assert_array_equal(np.asarray(observed).view(np.uint64), np.asarray(expected).view(np.uint64))


def main(provider, report):
    execution = check_execution()
    plugins.enable_discovery(False)
    plugins.load_library(str(provider.resolve()))
    cset = CorrectionSet.from_string(json.dumps({
        "schema_version": 3, "corrections": [
            correction_def("model", model(0.05)),
            correction_def("other", model(0.15)),
            correction_def("invalid", model(0.05, invalid=True)),
        ],
    }))
    primary, other, invalid = (cset[name] for name in ("model", "other", "invalid"))
    buffers = JetBuffers(provider)
    buffers.lib.jets_unsigned_pair_hits.argtypes = []
    buffers.lib.jets_unsigned_pair_hits.restype = ct.c_size_t
    hits = buffers.lib.jets_unsigned_pair_hits
    args = vector([80000, 120000, 250000, 500000])
    evidence = {}

    def chain(label, requests, expected_hits, rows=4):
        control = [node.evaluate(*a, state) for node, state, a in requests]
        with buffers.unsigned_sign_pairs(rows) as lease:
            for (node, state, a), expected in zip(requests, control, strict=True):
                equal(lease.evaluate(node.evaluate, *a, state), expected)
            assert hits() == expected_hits, (label, hits(), expected_hits)
            evidence[label] = expected_hits
        assert hits() == 0

    for family in ("shared", "bflavour", "resolution"):
        for pseudo in ("", "-pd"):
            names = [family + pseudo + "/" + name for name in ("one", "two")]
            chain(family + pseudo + "/canonical", [(primary, name, args) for name in names], 2)
            chain(family + pseudo + "/reverse", [(primary, name, args) for name in reversed(names)], 2)
    chain("component-validity", [(primary, f"guard/{s}", args) for s in ("one", "two")], 1)
    chain("file-validity", [(invalid, f"shared/{s}", args) for s in ("one", "two")], 0)
    chain("multi-scale ordered fallback", [(primary, f"multiscale/{s}", args) for s in ("one", "two")], 0)
    chain("same sign fallback", [(primary, "shared/one", args)] * 2, 0)
    chain("non-unit sign fallback", [(primary, "non-unit", args), (primary, "shared/one", args)], 0)
    chain("two model isolation", [(primary, "shared/one", args), (other, "shared/two", args)], 0)
    chain("pseudo boundary", [(primary, "shared/one", args), (primary, "shared-pd/two", args)], 0)
    chain("nonadjacent subset", [(primary, s, args) for s in ("shared/one", "NOSYS", "shared/two")], 0)
    chain("oversized scope", [(primary, f"shared/{s}", args) for s in ("one", "two")], 0, rows=65537)
    chain("row count mismatch", [(primary, f"shared/{s}", args) for s in ("one", "two")], 0, rows=3)

    # The low-level evalv entry borrows these contiguous NumPy columns while
    # executing. The trial must own its saved typed row and observe mutations
    # on the next call. Change a physics input for row 0 and leave row 1 intact.
    original = primary._base.evalv(*args, "shared/one")
    expected = primary._base.evalv(*vector([160000, 120000, 250000, 500000]), "shared/two")
    with buffers.unsigned_sign_pairs(4) as lease:
        equal(lease.evaluate(primary._base.evalv, *args, "shared/one"), original)
        args[0][0] = 160000.0
        equal(lease.evaluate(primary._base.evalv, *args, "shared/two"), expected)
        assert hits() == 1
    evidence["borrowed physics mutation"] = 1

    different = vector([80000, 120000, 250000, 500000])
    different[4][0] = -0.0
    nan = np.array([0x7fc00002], dtype=np.uint32).view(np.float32).astype(np.float64)[0]
    args = vector([80000, 120000, 250000, 500000])
    args[4][1] = np.array([0x7fc00001], dtype=np.uint32).view(np.float32).astype(np.float64)[0]
    different[4][1] = nan
    chain("signed zero and NaN payload guards",
          [(primary, "shared/one", args), (primary, "shared/two", different)], 0)

    original = vector([80000, 120000, 250000, 500000])
    reorder = [value.copy() if isinstance(value, np.ndarray) else value for value in original]
    reorder[0][0], reorder[0][1] = reorder[0][1], reorder[0][0]
    chain("row order mismatch", [(primary, "shared/one", original), (primary, "shared/two", reorder)], 0)
    empty = [value[:0] if isinstance(value, np.ndarray) else value for value in original]
    chain("empty chunk", [(primary, "shared/one", empty), (primary, "shared/two", empty)], 0, rows=0)

    rejected = list(original)
    rejected[7] = original[7].copy()
    rejected[7][1] = 2
    expected = primary.evaluate(*original, "shared/two")
    with buffers.unsigned_sign_pairs(4) as lease:
        lease.evaluate(primary.evaluate, *original, "shared/one")
        try:
            lease.evaluate(primary.evaluate, *rejected, "shared/two")
        except (ValueError, RuntimeError) as error:
            assert "zero or one" in str(error)
        else:
            raise AssertionError("Invalid presence flag reached the group")
        prior_hits = hits()
        equal(lease.evaluate(primary.evaluate, *original, "shared/two"), expected)
        assert hits() == prior_hits
    evidence["partial exception clears pending state"] = True

    with buffers.unsigned_sign_pairs(4) as lease:
        lease.evaluate(primary.evaluate, *original, "shared/one")
        lease.discard()
        equal(lease.evaluate(primary.evaluate, *original, "shared/two"), expected)
        assert hits() == 0
    evidence["explicit bypass breaks pair"] = True

    with buffers.unsigned_sign_pairs(4) as lease:
        lease.evaluate(primary.evaluate, *original, "shared/one")
        other.evaluate(*original, "shared/one")
        equal(lease.evaluate(primary.evaluate, *original, "shared/two"), expected)
        assert hits() == 0
    evidence["unscoped native interleaving breaks pair"] = True

    scalar_rows = [
        [value[i].item() if isinstance(value, np.ndarray) else value for value in original]
        for i in range(4)
    ]
    requested = [(primary if i < 2 else other, row) for i, row in enumerate(scalar_rows)]
    independent = [node.evaluate(*row, "shared/one") for node, row in requested]
    with buffers.unsigned_sign_pairs(4) as lease:
        lease.call("jets_unsigned_pair_begin_state")
        try:
            for (node, row), expected_row in zip(requested, independent, strict=True):
                equal(node.evaluate(*row, "shared/one"), expected_row)
        except BaseException:
            lease.call("jets_unsigned_pair_end_state", 0)
            raise
        lease.call("jets_unsigned_pair_end_state", 1)
        equal(lease.evaluate(primary.evaluate, *original, "shared/two"), expected)
        assert hits() == 0
    evidence["mixed model rows invalidate a state window"] = True

    status = primary.evaluate(*original, "shared/two")
    assert status[:, 4].tolist()[:2] == [2.0, 2.0]
    assert status[2, 5] == 0 and status[3, 4] == 0 and status[3, 5] == 1
    assert status[0, 6:8].tolist() == [1.0, 1.0] and status[1, 6:8].tolist() == [0.0, 1.0]
    assert np.all(primary.evaluate(*original, "shared-pd/two")[:, 7] == 0)
    scalar = [value[0].item() if isinstance(value, np.ndarray) else value for value in original]
    equal(primary.evaluate(*scalar, "shared/two"), expected[0])
    evidence["ordinary and pseudo status availability"] = True

    with report.open("x") as output:
        json.dump({"passed": True, "affinity": execution, "runtime": native_runtime(provider),
                   "cases": evidence}, output, indent=2)
        output.write("\n")
    print(f"{len(evidence)} exact CorrectionLib sign-pair fixture cases passed")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m tools.test_unsigned_sign_pairs PROVIDER NEW-RECEIPT")
    main(*map(Path, sys.argv[1:]))
