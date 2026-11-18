# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = [
#   "awkward==2.15.0",
#   "hist==2.12.1",
#   "numpy==2.5.5",
#   "typer==0.29.1",
#   "uproot==5.7.7",
# ]
# ///
"""Read original PHYSLITE, run shared CorrectionLib jet corrections, and fill pT histograms.

Use this campaign's CorrectionLib wheel (uv run --with /path/to/wheel.whl) and
native provider. Pin the process to one CPU with taskset. Stored controls need
neither a provider nor a payload.
"""

import os

# Set these before numpy, ROOT, or any library that might create a worker pool.
for key in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "RAYON_NUM_THREADS", "ROOT_MAX_THREADS",
):
    os.environ[key] = "1"

from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Annotated

import awkward as ak
import hist
import numpy as np
import typer
import uproot
from uproot.source.file import MemmapSource
from uproot.source.futures import TrivialExecutor

CHECKOUT = Path(__file__).resolve().parents[1]
if not __package__ and str(CHECKOUT) not in sys.path:
    sys.path.insert(0, str(CHECKOUT))

from columnar.inputs import (
    COLLECTIONS, LARGE_LABEL, LARGE_OBSERVABLES, P4_FIELDS, R4_TRUTH, R10_TRUTH, SMALL_RHO,
    branch_plan, event_rows, flatten, offsets, optional_event, optional_object,
    pick_feature, read_p4, resolve_branches, sampling_features, track_counts,
)

VALID = np.uint32(0xFFFFFFFF)
OUT_OF_VALIDITY = np.uint32(0xFFFFFFFE)
STAGES = ("stored", "const", "pileup", "etaJES", "gsc", "jms", "calib", "uncert", "final")
EVENT_FIELDS = {"eventSF"}
WEIGHT_DIRECTIONS = {"JET_NNJvtEfficiency__1up": 1, "JET_NNJvtEfficiency__1down": -1}


class Scope(str, Enum):
    small = "small"
    large = "large"
    both = "both"


class Mode(str, Enum):
    stored = "stored"
    nominal = "nominal"
    full = "full"


@dataclass(frozen=True)
class Frame:
    prefix: str
    stage: str
    state: str
    counts: np.ndarray
    fields: dict[str, np.ndarray]

    def name(self, field):
        return f"{self.prefix}_{field}_{self.state}"


def check_execution():
    process = Path("/proc/self/status").read_text()
    state = dict(line.split(":", 1) for line in process.splitlines()
                 if line.startswith(("Threads:", "Cpus_allowed_list:")))
    threads = int(state["Threads"])
    cpus = state["Cpus_allowed_list"].strip()
    if threads != 1 or not cpus.isdecimal():
        raise RuntimeError(f"Expected one Linux thread and one assigned CPU, observed Threads={threads}, "
                           f"Cpus_allowed_list={cpus}. Launch with taskset -c <cpu>.")
    return {"threads": threads, "cpu": int(cpus)}


def selected_collections(scope):
    if scope == Scope.small:
        return ("sr",)
    if scope == Scope.large:
        return ("lr",)
    return ("sr", "lr")


def select_states(mode, manifest, selected):
    states = {
        prefix: (list(manifest[f"{prefix}_histogram"]) if mode == Mode.full else ["NOSYS"])
        for prefix in selected
    }
    for prefix, names in states.items():
        if not names or names[0] != "NOSYS" or len(set(names)) != len(names):
            raise ValueError(f"{prefix} manifest requires unique states and NOSYS first")
    return states


def native_runtime(provider):
    import correctionlib

    package = Path(correctionlib.__file__).resolve().parent
    expected = (package / "lib/libcorrectionlib.so").resolve(strict=True)
    mapped = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith("/"):
            mapped.add(Path(parts[5]).resolve())
    cores = sorted(path for path in mapped if path.name.startswith("libcorrectionlib.so"))
    if cores != [expected]:
        raise RuntimeError(f"Expected exactly one CorrectionLib core mapped from {package}, found {cores}")
    provider_path = provider.resolve(strict=True)
    if provider_path not in mapped:
        raise RuntimeError(f"Requested jet provider is absent from /proc/self/maps: {provider_path}")
    used = ("libcorrectionlib.so", "libJetCalibToolsLib", "libJetUncertaintiesLib", "liblwtnn")
    return {"core": str(expected), "provider": str(provider_path),
            "mapped": sorted(str(path) for path in mapped if path.name.startswith(used) or
                             (path.parent == package and path.name.startswith("_core.cpython")))}


def load_corrections(provider, payload, need_transport=True):
    from correctionlib import CorrectionSet, plugins

    resolved = provider.resolve(strict=True)
    plugins.enable_discovery(False)
    plugins.load_library(str(resolved))
    nodes = CorrectionSet.from_file(str(payload))
    native_runtime(resolved)
    buffers = None
    if need_transport:
        from jet_transport import JetBuffers

        buffers = JetBuffers(resolved)
    check_execution()
    return nodes, buffers


def ff_states(provenance, stage, mode, states):
    if mode != Mode.full or stage not in ("uncert", "final"):
        return frozenset()
    if provenance is None:
        raise ValueError("Full evaluation requires the bundle provenance JSON")
    sources = json.loads(provenance.read_text())["sources"]
    ff_names = frozenset()
    if "lr" in states:
        ff_names = frozenset(
            name for component in sources["large_jmr"]["components"] for name in component["states"]
        )
    for prefix, names in states.items():
        collection = "SmallR" if prefix == "sr" else "LargeR"
        uncertainty_names = set(sources["uncertainties"]["collections"][collection]["states"])
        secondary_names = ff_names if prefix == "lr" else set(WEIGHT_DIRECTIONS)
        if uncertainty_names & secondary_names or set(names) != {"NOSYS", *uncertainty_names, *secondary_names}:
            raise ValueError(f"{prefix} manifest and payload systematic dispatch disagree")
    return ff_names


def p4_fields(stage, values):
    return {f"{stage}{field}": values[:, index] for index, field in enumerate(P4_FIELDS)}


def status_bits(status, present, describe):
    has_status = np.asarray(present, dtype=bool)
    if np.any(has_status & ((status != 1) & (status != 2))):
        raise ValueError(f"{describe} returned a release correction error")
    return np.where(has_status, np.where(status == 2, VALID, OUT_OF_VALIDITY), 0).astype(np.uint32)


def event_inputs(source, events, nodes, buffers):
    vertex_types = source["PrimaryVertices", "vertexType"]
    counts = np.asarray(ak.to_numpy(ak.num(vertex_types, axis=1)), dtype=np.int64)
    mu = np.asarray(ak.to_numpy(source["EventInfo", "averageInteractionsPerCrossing"]), dtype=np.float64)
    pv, has_pv = optional_event(source, "PVIndex", events, np.int64, 0)
    npv, has_npv = optional_event(source, "NPV", events, np.float64, 0)
    with buffers.vertices(offsets(counts), flatten(vertex_types, np.int64), track_counts(source)):
        prepared = nodes["jet_event_inputs"].evaluate(
            np.arange(events, dtype=np.int64), pv, has_pv, npv, has_npv)
    return mu, prepared


def truth_context(source, collection, buffers):
    counts = np.asarray(ak.to_numpy(ak.num(source[collection, "pt"], axis=1)), dtype=np.int64)
    coordinates = [flatten(source[collection, coordinate], np.float64) for coordinate in P4_FIELDS]
    return buffers.truth(offsets(counts), *coordinates)


def label_original_jets(source, selected, nodes, buffers):
    labels = {}
    with truth_context(source, R4_TRUTH, buffers):
        for prefix in selected:
            collection = COLLECTIONS[prefix]
            counts = np.asarray(ak.to_numpy(ak.num(source[collection, "pt"], axis=1)), dtype=np.int64)
            labels[prefix] = nodes["jet_pileup_truth"].evaluate(
                *read_p4(source, collection).T, event_rows(counts))
    return labels


def uncertainty_inputs(source, collection, calibration, rows, events):
    total = len(calibration)
    mu, prepared = events
    parton, parton_present = optional_object(source, collection, "PartonTruthLabelID", total, np.int64, 0)
    label, label_present = optional_object(source, collection, LARGE_LABEL, total, np.int64, 0)
    has_calo = [(collection, f"JetJMSScaleMomentumCalo_{coordinate}") in source for coordinate in P4_FIELDS]
    if any(has_calo) and not all(has_calo):
        raise ValueError(f"Incomplete named Calo four-vector for {collection}")
    calo = read_p4(source, collection, "JetJMSScaleMomentumCalo_") if all(has_calo) else np.zeros((total, 4))
    arguments = (*calibration.T, mu[rows], prepared[rows, 1], 1, 1,
                 parton, parton_present, label, label_present, *calo.T, int(all(has_calo)))
    return arguments, label, label_present


def small_jvt_inputs(source, collection, rows, prepared):
    jvt_vertex = prepared[rows, 3].astype(np.int64)
    vertex_kind = prepared[rows, 4].astype(np.int64)
    track_sum, available = pick_feature(source, collection, "SumPtTrkPt500", jvt_vertex, np.float64)
    per_jet_vectors = ak.flatten(source[collection, "SumPtTrkPt500"], axis=1)
    empty = np.asarray(ak.to_numpy(ak.num(per_jet_vectors, axis=1) == 0))
    feature_status = np.where(available, 1, np.where(empty, 0, -1)).astype(np.int64)
    jvf = flatten(source[collection, "JVFCorr"], np.float64)
    scores, has_scores = optional_object(source, collection, "NNJvt", len(rows), np.float64, -1)
    passes, has_pass = optional_object(source, collection, "NNJvtPass", len(rows), np.int64, 0)
    return track_sum, jvf, vertex_kind, feature_status, scores, has_scores, passes, has_pass


def small_jvt_selection(nodes, p4, features, state):
    track, jvf, kind, feature_status, inherited_score, has_score, inherited_pass, has_pass = features
    pt, eta = p4[:, 0], p4[:, 1]
    updated = nodes["small_nnjvt_update"].evaluate(pt, eta, track, jvf, kind, feature_status)
    if np.any(updated[:, 3] < 0):
        raise ValueError(f"NNJvt update returned an error in {state}")
    written = updated[:, 3] == 1
    score = np.where(written, updated[:, 0], inherited_score)
    score_present = np.where(written, 1, has_score).astype(np.int64)
    pass_decorator = np.where(written, updated[:, 2], inherited_pass).astype(np.int64)
    if not has_pass and np.any(~written):
        raise ValueError(f"Skipped NNJvt update with no inherited NNJvtPass in {state}")
    selection = nodes["small_nnjvt_select"].evaluate(pt, eta, score, score_present)
    if np.any(selection[:, 1] != 1):
        raise ValueError(f"NNJvt selection returned an error in {state}")
    return score, pass_decorator, selection[:, 0].astype(np.int64)


def small_sf(nodes, buffers, p4, selection, truth, counts, state, diagnose):
    value = nodes["small_nnjvt_sf"].evaluate(
        p4[:, 0], p4[:, 1], selection, truth, WEIGHT_DIRECTIONS.get(state, 0), 1)
    if np.any((value[:, 1] != 1) & (value[:, 1] != 2)):
        raise ValueError(f"Unexpected NNJvt SF status in {state}")
    fields = {"jvtEfficiency": value[:, 0], "hist_weight": value[:, 2]}
    if diagnose:
        fields["jvtValidity"] = np.where(value[:, 1] == 1, VALID, OUT_OF_VALIDITY)
    # The stock workflow also computes the per-event product. The requested
    # per-jet histogram uses the raw jet weight output instead.
    with buffers.weights(offsets(counts), value[:, 2]):
        event_sf = nodes["jet_event_sf"].evaluate(np.arange(len(counts), dtype=np.int64))
    if diagnose:
        fields["eventSF"] = event_sf
    return fields


def evaluate_chunk(source, events, selected, mode, states, ff_names, nodes, buffers,
                   diagnose=False, through="final", prepare_large_truth=False):
    stages_only = through != "final"
    if mode == Mode.stored or through == "stored":
        for prefix in selected:
            collection = COLLECTIONS[prefix]
            counts = np.asarray(ak.to_numpy(ak.num(source[collection, "pt"], axis=1)), dtype=np.int64)
            fields = (p4_fields("stored", read_p4(source, collection)) if diagnose
                      else {"p4pt": flatten(source[collection, "pt"], np.float64)})
            yield Frame(prefix, "stored" if diagnose else "final", "NOSYS", counts, fields)
        return

    if ("PrimaryVertices", "vertexType") in source:
        prepared = event_inputs(source, events, nodes, buffers)
    else:
        prepared = None
    pileup_labels = label_original_jets(source, selected, nodes, buffers) if not stages_only else {}

    for prefix in selected:
        collection = COLLECTIONS[prefix]
        counts = np.asarray(ak.to_numpy(ak.num(source[collection, "pt"], axis=1)), dtype=np.int64)
        rows = event_rows(counts)
        stored = read_p4(source, collection)
        constituent = read_p4(source, collection, "JetConstitScaleMomentum_")
        if diagnose:
            yield Frame(prefix, "stored", "NOSYS", counts, p4_fields("stored", stored))
            yield Frame(prefix, "const", "NOSYS", counts, p4_fields("const", constituent))
        if through == "const":
            continue

        detector_eta = flatten(source[collection, "DetectorEta"], np.float64)
        if prefix == "sr":
            mu, event_vertex = prepared
            rho = np.asarray(ak.to_numpy(source[SMALL_RHO, "Density"]), dtype=np.float64)
            area = flatten(source[collection, "ActiveArea4vec_pt"], np.float64)
            corrected = nodes["small_pileup"].evaluate(
                *constituent.T, area, rho[rows], mu[rows], event_vertex[rows, 0].astype(np.int64))
            if diagnose:
                yield Frame(prefix, "pileup", "NOSYS", counts, p4_fields("pileup", corrected))
            if through == "pileup":
                continue
        else:
            corrected = constituent

        corrected = nodes["small_eta_jes" if prefix == "sr" else "large_eta_jes"].evaluate(*corrected.T, detector_eta)
        if diagnose:
            yield Frame(prefix, "etaJES", "NOSYS", counts, p4_fields("etaJES", corrected))
        if through == "etaJES":
            continue

        if prefix == "sr":
            indexes = prepared[1][rows, 2].astype(np.int64)
            sum_pfo, pfo_present = pick_feature(source, collection, "SumPtChargedPFOPt500", indexes, np.float64)
            ntrk, ntrk_present = pick_feature(source, collection, "NumTrkPt1000", indexes, np.int64)
            width, width_present = pick_feature(source, collection, "TrackWidthPt1000", indexes, np.float64)
            if not np.all(pfo_present & ntrk_present & width_present):
                raise ValueError("Small-R GSC lacks a stored feature at the resolved primary-vertex index")
            energies = sampling_features(source, collection, (3, 7, 12, 18))
            corrected = nodes["small_gsc"].evaluate(
                *corrected.T, detector_eta, *constituent.T, sum_pfo, ntrk, width, *energies)
            if diagnose:
                yield Frame(prefix, "gsc", "NOSYS", counts, p4_fields("gsc", corrected))
            if through == "gsc":
                continue
        else:
            corrected = nodes["large_jms"].evaluate(*corrected.T, detector_eta)
            if diagnose:
                yield Frame(prefix, "jms", "NOSYS", counts, p4_fields("jms", corrected))
            if through == "jms":
                continue
        if diagnose:
            yield Frame(prefix, "calib", "NOSYS", counts, p4_fields("calib", corrected))
        if through == "calib":
            continue

        arguments, large_truth_label, large_label_present = uncertainty_inputs(
            source, collection, corrected, rows, prepared)
        uncertainty = nodes["small_uncertainty" if prefix == "sr" else "large_uncertainty"]
        validity_field = "jesJerValidity" if prefix == "sr" else "jesJERJMSValidity"
        nominal = None
        nominal_jvt = None
        features = (small_jvt_inputs(source, collection, rows, prepared[1])
                    if prefix == "sr" and not stages_only else None)
        context = (truth_context(source, R10_TRUTH, buffers)
                   if prefix == "lr" and (ff_names or prepare_large_truth) and not stages_only else nullcontext())
        with context:
            for state in states[prefix]:
                is_weight_only = prefix == "sr" and state in WEIGHT_DIRECTIONS
                is_ff = prefix == "lr" and state in ff_names
                if stages_only and (is_weight_only or is_ff):
                    continue
                if is_ff or is_weight_only:
                    varied = nominal
                else:
                    varied = uncertainty.evaluate(*arguments, state)
                    if state == "NOSYS":
                        nominal = varied
                valid_combined = status_bits(varied[:, 4], varied[:, 5], f"{prefix} {state} uncertainty")
                p4 = varied[:, :4]
                if diagnose and not (is_ff or is_weight_only):
                    intermediate = p4_fields("uncert", p4)
                    if stages_only:
                        intermediate[validity_field] = valid_combined
                    yield Frame(prefix, "uncert", state, counts, intermediate)
                if stages_only:
                    continue

                fields = {}
                if prefix == "sr":
                    truth = pileup_labels[prefix][:, 0].astype(np.int64)
                    if is_weight_only:
                        jvt = nominal_jvt
                    else:
                        jvt = small_jvt_selection(nodes, p4, features, state)
                        if state == "NOSYS":
                            nominal_jvt = jvt
                    score, pass_decorator, selection = jvt
                    fields.update(small_sf(nodes, buffers, p4, selection, truth, counts, state, diagnose))
                    if diagnose and not is_weight_only:
                        fields.update(p4_fields("p4", p4))
                        fields.update({validity_field: valid_combined, "nnjvt": score,
                                       "nnjvtPass": pass_decorator, "jvtSelection": selection, "isJvtHS": truth})
                    elif diagnose:
                        fields.pop("jvtValidity", None)
                else:
                    jmr_validity = np.zeros(len(p4), dtype=np.uint32)
                    if is_ff:
                        ff = nodes["large_jmr"].evaluate(*p4.T, large_truth_label, large_label_present, state, rows)
                        p4 = ff[:, :4]
                        jmr_validity = status_bits(ff[:, 4], ff[:, 5], f"{prefix} {state} JMR")
                    selection = nodes["large_selection"].evaluate(*p4.T)
                    fields["selected"] = selection[:, 2].astype(bool)
                    if diagnose:
                        fields.update(p4_fields("p4", p4))
                        fields.update({validity_field: valid_combined, "jmrValidity": jmr_validity,
                                       "selectPtEta": selection[:, 0], "selectMass": selection[:, 1]})
                        fields.update({name: flatten(source[collection, branch])
                                       for name, branch in LARGE_OBSERVABLES.items()})

                fields["hist_pt"] = p4[:, 0]
                if diagnose and not is_weight_only:
                    fields["index"] = np.asarray(ak.to_numpy(
                        ak.flatten(ak.local_index(source[collection, "pt"], axis=1), axis=1)), dtype=np.uint32)
                yield Frame(prefix, "final", state, counts, fields)


def create_histogram(edges, states):
    return hist.Hist(hist.axis.Variable(edges, name="pt"),
                     hist.axis.StrCategory(states, name="state", overflow=False),
                     storage=hist.storage.Weight())


def chunk_bounds(entries, n, size):
    events = entries if n is None else min(entries, n)
    return events, ((start, min(events, start + size)) for start in range(0, events, size))


def main(
    original: Path,
    output: Path,
    provider: Annotated[Path | None, typer.Option(help="Shared native CorrectionLib provider")] = None,
    payload: Annotated[Path | None, typer.Option(help="Generated schema-v3 correction set")] = None,
    provenance: Annotated[Path | None, typer.Option(help="Payload provenance JSON")] = None,
    name_manifest: Annotated[Path, typer.Option("--states")] = CHECKOUT / "config/jet-states.json",
    histogram_spec: Annotated[Path, typer.Option("--histogram-spec")] = CHECKOUT / "config/jet-histogram.json",
    scope: Scope = Scope.both,
    mode: Mode = Mode.nominal,
    repeat: Annotated[int, typer.Option("-n", "--repeat", min=1)] = 1,
    n: Annotated[int | None, typer.Option("-e", "--events", min=0)] = None,
    chunk: Annotated[int | None, typer.Option("--chunk", min=1)] = None,
    matched_read: Annotated[bool, typer.Option(
        "--matched-read", help="Read the full calibrated input and prepare common truth for nominal timing controls",
    )] = False,
):
    """Histogram jets directly from the original PHYSLITE CollectionTree."""
    check_execution()
    if output.exists():
        raise FileExistsError(output)
    selected = selected_collections(scope)
    mode_name = mode.value
    manifest = json.loads(name_manifest.read_text())
    spec = json.loads(histogram_spec.read_text())
    states = select_states(mode_name, manifest, selected)
    edges = np.asarray(spec["bins_gev"], dtype=np.float64)
    positions = {prefix: {state: index for index, state in enumerate(names)} for prefix, names in states.items()}
    if mode != Mode.stored:
        if provider is None or payload is None:
            raise typer.BadParameter("calibrated mode requires --provider and --payload")
        nodes, buffers = load_corrections(provider, payload)
    else:
        nodes = buffers = None
    provenance = provenance or (payload.with_name("provenance.json") if payload else None)
    ff_names = ff_states(provenance, "final", mode_name, states)

    matched_nominal = matched_read and mode == Mode.nominal
    plan = branch_plan(selected, Mode.full if matched_nominal else mode_name)
    first_pass = None
    first_objects = None
    file_uuid = None
    branch_digest = None
    event_visits = 0
    object_visits = {prefix: 0 for prefix in states}
    for pass_number in range(1, repeat + 1):
        h = {prefix: create_histogram(edges, names) for prefix, names in states.items()}
        counts = {prefix: np.zeros(len(names), dtype=np.int64) for prefix, names in states.items()}
        objects = {prefix: 0 for prefix in states}

        with uproot.open(original, handler=MemmapSource, num_fallback_workers=1,
                         decompression_executor=TrivialExecutor(),
                         interpretation_executor=TrivialExecutor()) as root_file:
            this_uuid = str(root_file.file.uuid)
            if file_uuid is not None and this_uuid != file_uuid:
                raise ValueError("Original PHYSLITE identity changed between complete passes")
            file_uuid = this_uuid
            tree = root_file["CollectionTree"]
            branches = resolve_branches(tree, plan)
            requested = sorted({name for name in branches.values() if name is not None})
            digest = sha256("\n".join(requested).encode("utf-8")).hexdigest()
            if branch_digest is not None and digest != branch_digest:
                raise ValueError("Resolved PHYSLITE branches changed between complete passes")
            branch_digest = digest
            events, ranges = chunk_bounds(tree.num_entries, n, chunk or spec["chunk_events"])
            for start, stop in ranges:
                arrays = tree.arrays(requested, entry_start=start, entry_stop=stop, how=dict, library="ak")
                source = {label: arrays[name] for label, name in branches.items() if name is not None}
                check_execution()
                for frame in evaluate_chunk(source, stop - start, selected, mode_name, states, ff_names, nodes, buffers,
                                            prepare_large_truth=matched_nominal):
                    if frame.state == "NOSYS":
                        objects[frame.prefix] += int(frame.counts.sum())
                    pt = frame.fields.get("hist_pt", frame.fields.get("p4pt"))
                    selection = frame.fields.get("selected")
                    if selection is not None:
                        pt = pt[selection]
                    h[frame.prefix].fill(pt=pt / 1000.0, state=frame.state, weight=frame.fields.get("hist_weight"))
                    counts[frame.prefix][positions[frame.prefix][frame.state]] += len(pt)
                del source, arrays, frame, pt, selection
                check_execution()

        results = {"edges": edges, "events": events}
        for prefix, names in states.items():
            results[f"{prefix}_states"] = np.asarray(names)
            results[f"{prefix}_entries"] = counts[prefix]
            results[f"{prefix}_sumw"] = h[prefix].values(flow=True)
            results[f"{prefix}_sumw2"] = h[prefix].variances(flow=True)
            object_visits[prefix] += objects[prefix]
        event_visits += events
        if repeat > 1:
            if first_pass is None:
                first_pass = results
                first_objects = objects
            else:
                for field, original_value in first_pass.items():
                    np.testing.assert_array_equal(results[field], original_value,
                                                  err_msg=f"Repeated pass {pass_number}, field {field}")
                if objects != first_objects:
                    raise ValueError(f"Input object counts changed in pass {pass_number}")
        typer.echo(f"Pass {pass_number}/{repeat}, {events} events, "
                   + ", ".join(f"{prefix} {objects[prefix]} original jets" for prefix in states))

    with output.open("xb") as stream:
        np.savez(stream, **results)
    typer.echo(f"{original.name} ({file_uuid}), {len(requested)} input branches")
    typer.echo(f"Input branch SHA-256: {branch_digest}")
    if matched_read and mode != Mode.stored:
        typer.echo("Calibrated input plan: matched full-state reads and truth preparation")
    typer.echo(f"{events} unique events, {event_visits} event visits across {repeat} passes")
    for prefix, names in states.items():
        typer.echo(f"{prefix} {objects[prefix]} unique jets, {object_visits[prefix]} jet visits, "
                   f"{counts[prefix][0]} nominal entries, {len(names)} states")
        typer.echo(", ".join(names))
    typer.echo(f"Saved the final single-pass histogram to {output}")


if __name__ == "__main__":
    typer.run(main)
