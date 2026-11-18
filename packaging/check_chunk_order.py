"""Check installed chunk execution and event-order invariance from original PHYSLITE."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_visible(path):
    # Start the wheel's compiler runtime before any other imports bring in native packages.
    sys.dont_write_bytecode = True
    name = "_installed_visible_jet_script"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def physical_hash(branches):
    return hashlib.sha256("\n".join(sorted(branches)).encode()).hexdigest()


def order_worker(visible, original, authority, output, events):
    demo = load_visible(visible)
    demo.check_execution()
    sys.path.insert(0, str(ROOT))
    import awkward as ak
    import numpy as np
    from atlas_jet_demo.inputs import EVENT_KEYS, open_root, read_chunk
    from tools.compare_columnar import Diagnostics, reference_branches

    scope = demo.Scope.both
    selected = demo.selected_collections(scope)
    mode = demo.Mode.full
    provenance = demo.DATA / "provenance.json"
    installed = json.loads((demo.DATA / "build.json").read_text())
    states = demo.select_states(mode, json.loads((demo.DATA / "jet-states.json").read_text()), selected)
    ff_names = demo.ff_states(provenance, "final", mode, states)
    reuse = installed.get("demo_defaults", {}).get("reuse_nominal_nnjvt_update", False)
    known = {key: stage for key, stage in reference_branches(selected, states, ff_names, "final", mode).items()
             if stage == "final"}
    required = set(known) | set(EVENT_KEYS)
    passed_rows = {f"lr_passedRows_{state}": "passed_rows" for state in states["lr"]}
    schema = {**known, **{key: "identity" for key in EVENT_KEYS}, **passed_rows}
    split = events // 2
    ranges = [(0, split), (split, events)]
    with open_root(original) as source_file, open_root(authority) as reference_file:
        source, reference = (source_file["CollectionTree"], reference_file["analysis"])
        original_uuid = str(source_file.file.uuid)
        if reference.num_entries < events or source.num_entries < events:
            raise ValueError("Observer or original source ends before requested intervals")
        if absent := required.difference(reference.keys()):
            raise ValueError(f"Stock observer lacks {len(absent)} final branches: {sorted(absent)[:10]}")
        normal = demo.resolve_branches(source, demo.branch_plan(selected, mode))
        observed = demo.resolve_branches(source, demo.branch_plan(selected, mode, diagnostics=True))
        if any(observed[leaf] != path for leaf, path in normal.items()):
            raise ValueError("Diagnostic wrapper changes an installed calibration leaf")
        extra = {leaf: path for leaf, path in observed.items() if leaf not in normal}
        normal_read, extras_read, observer_read = [], [], []
        for first, stop in ranges:
            normal_read.append(read_chunk(source, normal, first, stop))
            extras_read.append(read_chunk(source, extra, first, stop))
            observer_read.append(reference.arrays(sorted(required), entry_start=first, entry_stop=stop,
                                                  how=dict, library="ak"))

    def merge(parts):
        return {key: ak.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}

    inputs = {**merge(normal_read), **merge(extras_read)}
    stock = merge(observer_read)
    orderings = {
        "source_order": np.arange(events, dtype=np.int64),
        "right_then_left_reversed": np.r_[np.arange(events - 1, split - 1, -1), np.arange(0, split)],
    }
    nodes, buffers = demo.load_corrections(demo.PROVIDER, demo.DATA / "jets.json")
    results = {}
    for name, ordering in orderings.items():
        diag = Diagnostics(schema)
        values = {key: val[ordering] for key, val in inputs.items()}
        stock_order = {key: val[ordering] for key, val in stock.items()}
        ordered_identity = {field: np.asarray(ak.to_numpy(values["EventInfo", field])) for field in EVENT_KEYS}
        ordered_identity["source_entry"] = ordering
        identities = {key: value.tolist() for key, value in ordered_identity.items()}
        inverse = np.argsort(ordering)
        hashes = {}

        def save_field(key, expected, actual, counts=None):
            if key in hashes:
                raise ValueError(f"Duplicate installed frame field {key} in {name}")
            diag.compare(key, expected, actual, ordered_identity, 0, counts)
            if counts is None:
                canonical = actual[inverse]
                geometry = b""
            else:
                canonical = demo.flatten(ak.unflatten(actual, counts)[inverse])
                geometry = np.ascontiguousarray(counts[inverse]).tobytes()
            encoded = hashlib.sha256()
            encoded.update(canonical.dtype.str.encode())
            encoded.update(canonical.shape[0].to_bytes(8, "little"))
            encoded.update(geometry)
            encoded.update(np.ascontiguousarray(canonical).tobytes())
            hashes[key] = encoded.hexdigest()

        for field in EVENT_KEYS:
            save_field(field, stock_order[field], ordered_identity[field])
        for frame in demo.evaluate_chunk(values, events, selected, mode, states, ff_names, nodes, buffers,
                                         diagnose=True, through="final", reuse_nominal_nnjvt_update=reuse):
            if frame.stage != "final":
                continue
            for field, actual in frame.fields.items():
                key = frame.name(field)
                if key in known:
                    save_field(key, stock_order[key], actual, None if field in demo.EVENT_FIELDS else frame.counts)
            if frame.prefix == "lr":
                p = frame.state
                accepted = ((stock_order[f"lr_selectPtEta_{p}"] == demo.VALID) &
                            (stock_order[f"lr_selectMass_{p}"] == demo.VALID))
                save_field(f"lr_passedRows_{p}", accepted, frame.fields["selected"], frame.counts)
        if missing := set(schema).difference(hashes):
            raise ValueError(f"Installed script did not expose {len(missing)} final fields: {sorted(missing)[:10]}")
        bad = {key: check for key, check in diag.checks.items()
               if check["exact_mismatches"] or check["shape_mismatch_events"]}
        results[name] = {
            "source_entries_and_event_ids": identities,
            "field_comparisons": sum(value["comparisons"] for value in diag.checks.values()),
            "fields": len(hashes),
            "normalized_field_hashes": hashes,
            "stock_mismatches": bad,
        }
        demo.check_execution()

    first, second = results.values()
    differing = [key for key, value in first["normalized_field_hashes"].items()
                 if value != second["normalized_field_hashes"][key]]
    normal_branches = sorted({path for path in normal.values() if path is not None})
    with visible.open("rb") as stream:
        script_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "passed": not differing and all(not case["stock_mismatches"] for case in results.values()),
        "original": str(original), "original_uuid": original_uuid,
        "stock_observer": str(authority),
        "installed_visible_source_sha256": script_sha,
        "execution": demo.check_execution(),
        "loaded_native": demo.native_runtime(demo.PROVIDER),
        "original_mapped_intervals": ranges,
        "order_scope": "Combined evaluator only. The same original rows are permuted in memory; no rewritten or reordered ROOT file is used.",
        "identical_calibration_input_branch_plan": normal_branches,
        "calibration_branch_sha256": physical_hash(normal_branches),
        "supplemental_identity_or_diagnostic_branches": sorted({path for path in extra.values() if path is not None}),
        "matched_production_settings": installed.get("demo_defaults"), "states": states,
        "runs": results, "bitwise_ordering_differences": differing,
    }
    demo.audit_runtime()
    _report(output, report)
    if not report["passed"]:
        raise ValueError(f"Installed event-order comparison failed; see {output}")


def _report(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "_order_worker":
    order_worker(*(Path(value) for value in sys.argv[2:6]), int(sys.argv[6]))
    sys.exit(0)


from typing import Annotated

import typer

from check_demo import CLEAN_ENV, digest, run
from compare_scoped_demo import (
    Scope, branch_digest, command, compare, installed_commands, installed_index,
    projection_receipt, require_cpu, run_leg,
)

APP = typer.Typer(add_completion=False)
FULL_EVENTS = 20_000
CHUNKS = (256, 1000, 4000)


def authoritative_original(path):
    current = path.resolve(strict=True)
    frozen = json.loads((ROOT / "config/jets-input.json").read_text())
    if current != Path(frozen["path"]).resolve(strict=True) or current.stat().st_size != frozen["bytes"]:
        raise ValueError("Use the original PHYSLITE file frozen in config/jets-input.json")
    return current, frozen


@APP.command()
def check(
    installation: Path,
    original: Path,
    observer: Path,
    reference_small_sample: Path,
    output: Path,
    order_events: Annotated[int, typer.Option(min=32, max=128)] = 64,
):
    """Check physical file-order histograms and reordered evaluator values against stock A."""
    require_cpu()
    original, dataset = authoritative_original(original)
    installation, observer, reference_small_sample, output = (
        path.resolve() for path in (installation, observer, reference_small_sample, output))
    commands, installed = installed_commands(installation)
    output.mkdir()
    inputs = {}
    for chunk in (7, 16, 32):
        run(command(commands["pip"], original, output / f"chunk-{chunk}.npz", Scope.both, "full", 32,
                    ["--chunk", str(chunk)]), output, f"chunk-{chunk}", 3)
        compare(reference_small_sample, output / f"chunk-{chunk}.npz")
        inputs[str(chunk)] = branch_digest(output / f"chunk-{chunk}.log")
    reordered = output / "evaluator-order.json"
    run([commands["pip"][0], str(Path(__file__).resolve()), "_order_worker", commands["pip"][1],
         original, observer, reordered, order_events], output, "evaluator-order", 3,
        {"ATLAS_JET_LIBRARY_AUDIT": str(output / "evaluator-native-libraries.json")})
    report = json.loads(reordered.read_text())
    if (report["original_uuid"] != dataset["tfile_uuid"] or not report["passed"] or
            report["installed_visible_source_sha256"] != installed["script_sha256"] or
            any(branch != report["calibration_branch_sha256"] for branch in inputs.values())):
        raise ValueError("File-order CLI and evaluator-only check disagree on input identity or leaves")
    _report(output / "summary.json", {
        "passed": True, "installation": installed,
        "file_order_cli": {
            "scope": "both", "events": 32, "chunk_sizes": [7, 16, 32],
            "physical_branch_digests": inputs,
            "stock_histogram": {"path": str(reference_small_sample), "sha256": digest(reference_small_sample)},
            "histograms_exact": True,
        },
        "evaluator_only": {
            "events": order_events, "proof": str(reordered),
            "row_and_state_values_bitwise_invariant": True,
            "values_and_flags_exact_to_stock_observer": True,
        },
    })
    require_cpu()


@APP.command()
def screen(
    installation: Path,
    original: Path,
    reference: Path,
    output: Path,
    scope: Annotated[Scope | None, typer.Option(help="Screen one scope. By default, screen small, large and both.")] = None,
):
    """Screen physical chunk sizes on the original full file through the uninstrumented installed command."""
    require_cpu()
    original, dataset = authoritative_original(original)
    installation, reference, output = (path.resolve() for path in (installation, reference, output))
    commands, installed = installed_commands(installation)
    scopes = [scope] if scope else list(Scope)
    defaults = installed.get("demo_defaults")
    output.mkdir()
    index = installed_index(installation) if defaults and defaults["compact_payload_default_scopes"] else {}
    measurements = {}
    for chosen in scopes:
        number = list(Scope).index(chosen)
        reference_scope = "combined" if chosen == Scope.both else chosen.value
        expected = reference / f"full-file-pilot-{reference_scope}-002/01-full/histograms.npz"
        row = measurements[chosen.value] = {}
        for chunk in CHUNKS[number:] + CHUNKS[:number]:
            case = output / f"{chosen.value}-{chunk}"
            case.mkdir()
            histograms = case / "histograms.npz"
            process = run_leg(
                command(commands["pip"], original, histograms, chosen, "full", FULL_EVENTS, ["--chunk", str(chunk)]),
                CLEAN_ENV, case, "installed", event_source=original,
            )
            compare(expected, histograms)
            log = case / "installed.log"
            branch_hash = branch_digest(log)
            expected_identity = f"({dataset['tfile_uuid']})"
            if expected_identity not in log.read_text() or not process["source_mapping_observed"]:
                raise ValueError(f"Could not verify the timed process used the frozen file in {case}")
            if row and branch_hash != next(iter(row.values()))["branch_sha256"]:
                raise ValueError(f"Physical branch plan changed for {chosen.value}")
            requested_default = defaults and chosen.value in defaults["compact_payload_default_scopes"]
            observed = projection_receipt(log, index[chosen.value] if requested_default else None) if defaults else {}
            row[str(chunk)] = {
                "wall_s": process["wall_s"], "cpu_s": process["cpu_user_s"] + process["cpu_sys_s"],
                "peak_rss_kB": process["peak_rss_kb"], "branch_sha256": branch_hash,
                "projection": observed, "exact_stock_histograms": True,
                "source_mapping_observed": process["source_mapping_observed"],
                "stock_reference": {"path": str(expected), "sha256": digest(expected)},
                "job_receipt": str(case / "installed.json"),
            }
    _report(output / "summary.json", {
        "classification": "Single-pass chunk-size engineering screen; review guest receipts before interpreting times.",
        "installation": installed, "original": str(original), "original_uuid": dataset["tfile_uuid"],
        "events": FULL_EVENTS, "matched_read": True, "measurements": measurements,
    })
    require_cpu()


if __name__ == "__main__":
    APP()
