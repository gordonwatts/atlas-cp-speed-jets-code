"""Compare direct PHYSLITE jet outputs to a stock AnalysisBase reference observer."""

from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Annotated

# Support both python -m tools.compare_columnar and direct script invocation.
CHECKOUT = Path(__file__).resolve().parents[1]
if str(CHECKOUT) not in sys.path:
    sys.path.insert(0, str(CHECKOUT))

from columnar.jets import (
    EVENT_FIELDS, OUT_OF_VALIDITY, STAGES, VALID, WEIGHT_DIRECTIONS, Mode, Scope, check_execution, chunk_bounds,
    evaluate_chunk, ff_states, load_corrections, native_runtime, select_states, selected_collections,
)
from columnar.inputs import (
    EVENT_KEYS, LARGE_OBSERVABLES, P4_FIELDS, branch_plan, flatten, offsets,
    open_root, read_chunk, resolve_branches,
)

import awkward as ak
import numpy as np
import typer

UNOBSERVED_FRAME_FIELDS = {"hist_pt", "hist_weight", "selected"}


def active_collections(scope, stage):
    required = {"pileup": "sr", "gsc": "sr", "jms": "lr"}.get(stage)
    selected = selected_collections(scope)
    if required and required not in selected:
        raise ValueError(f"Stage {stage} does not apply to {scope.value}")
    return (required,) if required else selected


def stage_names(prefix, state, stage, ff_names):
    if stage in ("stored", "const", "pileup", "etaJES", "gsc", "jms", "calib"):
        if state != "NOSYS":
            return []
        return [f"{stage}{coordinate}" for coordinate in P4_FIELDS]
    weight_only = prefix == "sr" and state in WEIGHT_DIRECTIONS
    is_ff = prefix == "lr" and state in ff_names
    if stage == "uncert":
        return [] if weight_only or is_ff else [f"uncert{coordinate}" for coordinate in P4_FIELDS]
    p4 = [f"p4{coordinate}" for coordinate in P4_FIELDS]
    if prefix == "sr":
        weights = ["jvtEfficiency", "eventSF"]
        return weights if weight_only else [
            *p4, "index", *weights, "jesJerValidity", "nnjvt",
            "nnjvtPass", "jvtSelection", "jvtValidity", "isJvtHS",
        ]
    return [*p4, "index", "jesJERJMSValidity", "jmrValidity", "selectPtEta",
            "selectMass", *LARGE_OBSERVABLES]


def reference_branches(selected, states, ff_names, requested_stage, mode):
    branches = {}
    for prefix in selected:
        nominal_stages = ("stored", "const", "pileup", "etaJES", "gsc", "calib") if prefix == "sr" else (
            "stored", "const", "etaJES", "jms", "calib")
        if mode == Mode.stored:
            compare_stages = ("stored",)
        elif requested_stage == "final":
            compare_stages = (*nominal_stages, "uncert", "final")
        else:
            compare_stages = (requested_stage,)
        for stage in compare_stages:
            for state in states[prefix] if stage in ("uncert", "final") else ("NOSYS",):
                fields = stage_names(prefix, state, stage, ff_names)
                if fields and requested_stage == "uncert":
                    fields.append("jesJerValidity" if prefix == "sr" else "jesJERJMSValidity")
                for field in fields:
                    branches[f"{prefix}_{field}_{state}"] = stage
    return branches


def scalar(value):
    result = value.item() if isinstance(value, np.generic) else value
    if isinstance(result, float) and not math.isfinite(result):
        return str(result)
    return result


def exact_equal(reference, actual):
    equal = reference == actual
    if reference.dtype.kind == "f" or actual.dtype.kind == "f":
        equal |= np.isnan(reference) & np.isnan(actual)
    return equal


def event_identity(events, start, local_event):
    return {"entry": start + int(local_event),
            **{key: scalar(values[local_event]) for key, values in events.items()}}


class Diagnostics:
    def __init__(self, expected):
        self.checks = {
            key: {"stage": stage, "comparisons": 0, "exact_mismatches": 0, "shape_mismatch_events": 0,
                  "max_abs_difference": 0.0, "first": None}
            for key, stage in expected.items()
        }
        self.seen = Counter()

    def compare(self, key, authority, candidate, identities, start, counts=None):
        result = self.checks[key]
        self.seen[key] += 1
        if counts is None:
            expected = np.asarray(ak.to_numpy(authority))
            if expected.shape != candidate.shape:
                raise ValueError(f"Event-valued branch shape mismatch for {key} at entry {start}")
        else:
            old_counts = np.asarray(ak.to_numpy(ak.num(authority, axis=1)), dtype=np.int64)
            if old_counts.shape != counts.shape:
                raise ValueError(f"Event axis differs for {key} at entry {start}")
            different = np.flatnonzero(old_counts != counts)
            if different.size:
                result["shape_mismatch_events"] += int(len(different))
                if result["first"] is None:
                    idx = int(different[0])
                    result["first"] = {
                        **event_identity(identities, start, idx), "reason": "object count",
                        "A": int(old_counts[idx]), "C": int(counts[idx]),
                    }
                return
            expected = flatten(authority)
            if expected.shape != candidate.shape:
                raise ValueError(f"Flat candidate shape differs for {key} at entry {start}")

        result["comparisons"] += int(len(expected))
        bad = np.flatnonzero(~exact_equal(expected, candidate))
        result["exact_mismatches"] += int(len(bad))
        if not bad.size:
            return
        a = expected[bad].astype(np.float64)
        c = candidate[bad].astype(np.float64)
        finite = np.isfinite(a) & np.isfinite(c)
        result["max_abs_difference"] = max(
            result["max_abs_difference"], float(np.max(np.abs(a[finite] - c[finite]), initial=0.0)))
        if result["first"] is None:
            index = int(bad[0])
            if counts is not None:
                out_offsets = offsets(counts)
                event = int(np.searchsorted(out_offsets, index, side="right") - 1)
                obj = int(index - out_offsets[event])
            else:
                event, obj = index, None
            result["first"] = {
                **event_identity(identities, start, event), "original_jet_index": obj,
                "A": scalar(expected[index]), "C": scalar(candidate[index]),
            }


def histogram_diff(authority, candidate):
    with np.load(authority, allow_pickle=False) as expected, np.load(candidate, allow_pickle=False) as actual:
        if set(expected.files) != set(actual.files):
            raise ValueError("Path A and C histogram NPZ fields differ")
        checks = {}
        for key in expected.files:
            a, c = expected[key], actual[key]
            if a.shape != c.shape:
                checks[key] = {"A_shape": a.shape, "C_shape": c.shape, "passed": False}
                continue
            bad = np.argwhere(~exact_equal(a, c))
            if a.ndim == 0:
                mismatch = int(not bool(exact_equal(a, c)))
            else:
                mismatch = len(bad)
            first = None if not mismatch else tuple(bad[0].tolist())
            checks[key] = {"comparisons": int(a.size), "mismatches": mismatch, "passed": not mismatch,
                           "first_bin_or_field": first}
            if mismatch:
                checks[key].update(A=scalar(a[first]), C=scalar(c[first]))
    return checks


def main(
    original: Path,
    authority: Path,
    output: Path,
    provider: Annotated[Path | None, typer.Option()] = None,
    payload: Annotated[Path | None, typer.Option()] = None,
    provenance: Annotated[Path | None, typer.Option()] = None,
    states_file: Annotated[Path, typer.Option("--states")] = CHECKOUT / "config/jet-states.json",
    scope: Scope = Scope.both,
    mode: Mode = Mode.full,
    stage: Annotated[str, typer.Option(help="First-divergence isolation through one native stage")] = "final",
    n: Annotated[int | None, typer.Option("-n", "--events", min=1)] = None,
    chunk: Annotated[int, typer.Option("--chunk", min=1)] = 32,
    histogram_a: Annotated[Path | None, typer.Option("--histogram-a")] = None,
    histogram_c: Annotated[Path | None, typer.Option("--histogram-c")] = None,
    distributions: Annotated[Path | None, typer.Option(help="Write independent endpoint distributions in a new directory")] = None,
):
    """Record exact per-object, event, and status agreement for configured states."""
    execution = check_execution()
    if output.exists():
        raise FileExistsError(output)
    if stage not in STAGES or (mode == Mode.stored and stage not in ("stored", "final")):
        raise typer.BadParameter(f"Unsupported comparison stage {stage!r} for {mode.value}")
    if (histogram_a is None) != (histogram_c is None):
        raise typer.BadParameter("Supply both histograms or neither")
    selected = active_collections(scope, stage)
    manifest = json.loads(states_file.read_text())
    states = select_states(mode, manifest, selected)
    provenance = provenance or (payload.with_name("provenance.json") if payload else None)
    ff_names = ff_states(provenance, stage, mode, states)
    expected = reference_branches(selected, states, ff_names, stage, mode)
    diagnostic = Diagnostics({**expected, **{key: "identity" for key in EVENT_KEYS}})
    plan = branch_plan(selected, mode, diagnostics=True, stage=stage)

    recorder = None
    if distributions is not None:
        if mode == Mode.stored or stage != "final":
            raise typer.BadParameter("Endpoint distributions require a calibrated final comparison")
        if distributions.exists():
            raise FileExistsError(distributions)
        from columnar.distributions import EndpointDistributions

        recorder = EndpointDistributions(states, WEIGHT_DIRECTIONS, VALID)

    nodes = buffers = None
    if mode != Mode.stored and stage not in ("stored", "const"):
        if provider is None or payload is None:
            raise typer.BadParameter("Calibrated comparisons require --provider and --payload")
        need_transport = "sr" in selected or stage in ("uncert", "final")
        nodes, buffers = load_corrections(provider, payload, need_transport)
    failure = None
    events_seen = 0
    with open_root(original) as original_file, open_root(authority) as authority_file:
        src, ref = original_file["CollectionTree"], authority_file["analysis"]
        original_uuid = str(original_file.file.uuid)
        if ref.num_entries == 0 or ref.num_entries > src.num_entries:
            raise ValueError("Reference entry range is incompatible with PHYSLITE input")
        if n is not None and n > ref.num_entries:
            raise ValueError("Reference ends before the requested comparison interval")
        raw = resolve_branches(src, plan)
        required = set(expected) | set(EVENT_KEYS)
        absent = required.difference(ref.keys())
        if absent:
            raise ValueError(f"Reference does not expose {len(absent)} required branches: {sorted(absent)[:10]}")

        entries, ranges = chunk_bounds(ref.num_entries, n, chunk)
        for start, stop in ranges:
            try:
                original_data = read_chunk(src, raw, start, stop)
                reference = ref.arrays(sorted(required), entry_start=start, entry_stop=stop, how=dict, library="ak")
                identities = {field: np.asarray(ak.to_numpy(original_data["EventInfo", field]))
                              for field in EVENT_KEYS}
                for field, values in identities.items():
                    diagnostic.compare(field, reference[field], values, identities, start)
                seen_here = set()
                if recorder is not None:
                    recorder.begin_chunk()
                for frame in evaluate_chunk(original_data, stop - start, selected, mode, states, ff_names,
                                            nodes, buffers, diagnose=True, through=stage):
                    for field, values in frame.fields.items():
                        if field in UNOBSERVED_FRAME_FIELDS:
                            continue
                        key = frame.name(field)
                        if key not in expected:
                            continue
                        if key in seen_here:
                            raise ValueError(f"Duplicate columnar output for {key}")
                        seen_here.add(key)
                        diagnostic.compare(key, reference[key], values, identities, start,
                                           None if field in EVENT_FIELDS else frame.counts)
                    if recorder is not None and frame.stage == "final":
                        recorder.record(frame, reference)
                missing = set(expected) - seen_here
                if missing:
                    raise ValueError(f"Missing {len(missing)} columnar outputs: {sorted(missing)[:10]}")
                check_execution()
                events_seen = stop
            except (ValueError, RuntimeError, KeyError) as error:
                failure = {"entry_start": start, "entry_stop": stop, "message": str(error)}
                break

    histogram = (histogram_diff(histogram_a, histogram_c) if histogram_a is not None else None)
    failed = {key: check for key, check in diagnostic.checks.items()
              if check["exact_mismatches"] or check["shape_mismatch_events"]}
    by_stage = Counter(check["stage"] for check in failed.values())
    first_stage = next((name for name in ("identity", *STAGES) if by_stage[name]), None)
    complete = events_seen == entries
    passed = (failure is None and complete and not failed and
              (histogram is None or all(check["passed"] for check in histogram.values())))
    report = {
        "passed": passed, "scope": scope.value, "mode": mode.value, "stage_limit": stage,
        "source_phySLITE": str(original.resolve()), "source_file_uuid": original_uuid,
        "authority": str(authority.resolve()), "provider": str(provider.resolve()) if provider else None,
        "payload": str(payload.resolve()) if payload else None, "execution": execution,
        "runtime": native_runtime(provider) if nodes is not None else None,
        "entries_requested": entries, "entries_compared": events_seen,
        "states": states, "observer_branches_comparable": len(expected),
        "status_encoding": {"0": "absent", str(VALID): "valid", str(OUT_OF_VALIDITY): "out_of_validity"},
        "large_r_jmr_execution": (
            {state: "evaluated" if state in ff_names else "structurally_absent" for state in states["lr"]}
            if mode != Mode.stored and stage == "final" and "lr" in states else None
        ),
        "source_branches_read": sorted({value for value in raw.values() if value is not None}),
        "first_mismatching_stage": first_stage, "mismatching_branches_by_stage": dict(by_stage),
        "execution_failure": failure, "checks": diagnostic.checks, "histogram": histogram,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if recorder is not None:
        recorder.write(distributions, output, events_seen, ff_names)
    typer.echo(f"{events_seen}/{entries} events, {len(failed)} mismatching branches")
    if not passed:
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
