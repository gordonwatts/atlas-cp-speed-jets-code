"""Prepare independently observed jet coordinates and replay two active uncertainty tables."""

import json
import mmap
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Annotated

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exporters.payload_ranges import identify
sys.path.insert(0, str(ROOT / "tools/reference"))
from workflow import open_root

APP = typer.Typer(add_completion=False)
CASES = {
    "small": ("sr", "SmallR", "JET_EtaIntercalibration_NonClosure_0p2_PreRec", 16),
    "large": ("lr", "LargeR", "JET_EtaIntercalibration_NonClosure_R10_PreRec", 128),
}


def receipt(path: Path, report) -> None:
    with path.open("x") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")


def require_cpu():
    if os.sched_getaffinity(0) != {3} or len(list(Path("/proc/self/task").iterdir())) != 1:
        raise ValueError("Run on the reserved analysis CPU3 with one thread")


def table_bytes(canonical: Path, span: list[int], table: str) -> bytes:
    marker = json.dumps(table).encode() + b":{"
    with canonical.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        first = data.find(marker, *span)
        if first < 0 or data.find(marker, first + len(marker), span[1]) >= 0:
            raise ValueError(f"Expected one raw histogram object for {table}")
        first += len(marker) - 1
        stop = min(first + 65536, span[1])
        while True:
            try:
                _, characters = json.JSONDecoder().raw_decode(data[first:stop].decode("ascii"))
                return data[first:first + characters]
            except json.JSONDecodeError:
                if stop >= span[1]:
                    raise
                stop = min(first + 2 * (stop - first), span[1])


@APP.command()
def prepare(observer: Path, canonical: Path, partition: Path, output: Path) -> None:
    """Extract exact JSON histogram objects and paired pre-uncertainty observer coordinates."""
    require_cpu()
    import awkward as ak
    import numpy as np

    observer = observer.resolve(strict=True)
    canonical = canonical.resolve(strict=True)
    initialized = observer.with_name("initialized-tools.tsv")
    observer_sha = observer.with_name("reference.sha256").read_text().split()[0]
    manifest = json.loads((partition / "manifest.json").read_text())
    actual = identify(canonical)
    if any(manifest["canonical"][field] != value for field, value in actual.items()):
        raise ValueError("The source partition differs from the selected portable CorrectionSet")
    plan_file = canonical.with_name("plan.json")
    provenance_file = canonical.with_name("provenance.json")
    plan = json.loads(plan_file.read_text())
    provenance = json.loads(provenance_file.read_text())["sources"]["uncertainties"]["collections"]
    states = json.loads((ROOT / "config/jet-states.json").read_text())
    initialized_lines = initialized.read_text().splitlines()
    output.mkdir()
    records = {}
    with open_root(observer) as root_file:
        tree = root_file["analysis"]
        for name, (prefix, collection, component, passes) in CASES.items():
            correction = name + "_uncertainty"
            variation = component + "__1up"
            if variation not in states[f"{prefix}_p4" if name == "small" else "lr_histogram"]:
                raise ValueError(f"{variation} is absent from the executed state manifest")
            if not any(component in line and "\trecommended\t" in line for line in initialized_lines):
                raise ValueError(f"{component} is absent from the observed initialized recommendations")
            selected = provenance[collection]
            if selected["states"][variation]["basename"] != component:
                raise ValueError(f"Unexpected resolved variation {variation}")
            options, = plan[correction]["groups"][component]["components"]
            if (options["kind"], options["param"], options["mode"], options["scale"]) != (
                "table", "PtEta", "Full", "fourvec"
            ) or options["labels"] or options["validity"]:
                raise ValueError(f"Unexpected lookup options: {options}")
            table, = options["tables"]
            release_file = selected["histogram_file"]
            release_config = selected["recommendation"]
            for entry in (release_file, release_config):
                if identify(Path(entry["resolved_path"])) != {key: entry[key] for key in ("bytes", "sha256")}:
                    raise ValueError(f"Release source changed: {entry['resolved_path']}")
            wire_table = output / f"{name}-table.json"
            wire_table.write_bytes(table_bytes(canonical, manifest["nodes"][correction]["characters"], table))
            header = json.loads(wire_table.read_bytes())
            root_object = release_file["objects"][table]
            if header["class"] != root_object["class"] or len(header["axes"]) != 2:
                raise ValueError(f"Exported {table} is not the expected source histogram")
            fields = [f"{prefix}_calib{q}_NOSYS" for q in ("pt", "eta")]
            arrays = tree.arrays(fields, how=dict, library="ak")
            if not bool(ak.all(ak.num(arrays[fields[0]]) == ak.num(arrays[fields[1]]))):
                raise ValueError(f"Observer object boundaries disagree for {prefix}")
            points = np.column_stack([ak.to_numpy(ak.flatten(arrays[field])) for field in fields]).astype("<f4")
            raw = output / f"{name}-points-le-f32.bin"
            points.tofile(raw)
            records[name] = {
                "radius": name,
                "release_config": release_config,
                "release_histogram_file": {key: value for key, value in release_file.items() if key != "objects"},
                "table_key": table, "root_object": root_object,
                "correction": correction, "variation": variation,
                "options": options, "passes": passes,
                "mev_to_gev": selected["source_numerics"]["mev_to_gev"],
                "table_file": {"name": wire_table.name, **identify(wire_table)},
                "observer_branches": fields,
                "points": len(points),
                "points_file": {"name": raw.name, **identify(raw)},
                "coordinate_format": "Adjacent little-endian IEEE float32 pT [MeV] and signed eta per observer jet, in source entry order",
            }
        receipt(output / "sources.json", {
            "analysis_base": "25.2.117",
            "observer": {"path": str(observer), "bytes": observer.stat().st_size,
                         "sha256_from_sealed_capture": observer_sha,
                         "uuid": str(root_file.file.uuid), "entries": tree.num_entries},
            "initialized_tools": {"path": str(initialized), **identify(initialized)},
            "canonical_payload": {"path": str(canonical), **actual},
            "portable_plan": {"path": str(plan_file), **identify(plan_file)},
            "provenance": {"path": str(provenance_file), **identify(provenance_file)},
            "cases": records,
        })
    require_cpu()


def cpu_snapshot() -> dict:
    return {
        "cpu3": next(row for row in Path("/proc/stat").read_text().splitlines() if row.startswith("cpu3 ")),
        "io_pressure": Path("/proc/pressure/io").read_text(),
    }


@APP.command()
def run(prepared: Path, binary: Path, output: Path,
        rounds: Annotated[int, typer.Option(min=4)] = 7) -> None:
    """Prove exact table values then measure alternating paired lookups with both tables preloaded."""
    require_cpu()
    prepared = prepared.resolve(strict=True)
    binary = binary.resolve(strict=True)
    source = ROOT / "tools/reference/setup.sh"
    output = output.resolve()
    output.mkdir()
    records = json.loads((prepared / "sources.json").read_text())["cases"]
    summary = {}
    for name, case in records.items():
        inputs = [prepared / case[key]["name"] for key in ("table_file", "points_file")]
        for path, key in zip(inputs, ("table_file", "points_file"), strict=True):
            expected = {key2: case[key][key2] for key2 in ("bytes", "sha256")}
            if identify(path) != expected:
                raise ValueError(f"Frozen replay input changed: {path}")
        command = ["/bin/bash", "-c", 'source "$1" || exit; shift; exec "$@"', "replay", str(source), str(binary),
                   case["release_histogram_file"]["resolved_path"], case["table_key"], str(case["root_object"]["cycle"]),
                   *map(str, inputs), str(case["mev_to_gev"]), str(case["passes"]), str(rounds)]
        logfile = output / f"{name}.log"
        before = cpu_snapshot()
        with logfile.open("xb") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        after = cpu_snapshot()
        receipt(output / f"{name}-process.json", {"command": command, "exit": completed.returncode,
                                                   "before": before, "after": after})
        completed.check_returncode()
        lines = logfile.read_text().splitlines()
        proof, = [line.split("\t")[1:] for line in lines if line.startswith("proof\t")]
        if int(proof[0]) != case["points"] or proof[4] != case["root_object"]["class"]:
            raise ValueError("Proof row differs from the frozen observer or ROOT metadata")
        samples = []
        for row in lines:
            if row.startswith("sample\t"):
                _, backend, round_number, calls, wall, cpu, checksum = row.split("\t")
                samples.append({"backend": backend, "round": int(round_number), "calls": int(calls),
                                "wall_s": float(wall), "thread_cpu_s": float(cpu), "checksum": float(checksum)})
        expected_order = [backend for i in range(rounds) for backend in
                          (("release", "portable") if i % 2 == 0 else ("portable", "release"))]
        expected_calls = case["points"] * case["passes"]
        if ([item["backend"] for item in samples] != expected_order
                or any(item["round"] != i // 2 or item["calls"] != expected_calls for i, item in enumerate(samples))):
            raise ValueError("Unexpected paired replay order")
        retained = [item for item in samples if item["round"] > 0]
        rates = {backend: [item["wall_s"] * 1e9 / item["calls"] for item in retained if item["backend"] == backend]
                 for backend in ("release", "portable")}
        quality = {}
        for backend, values in rates.items():
            median = statistics.median(values)
            cv = statistics.stdev(values) / statistics.mean(values)
            deviation = max(abs(v - median) / median for v in values)
            quality[backend] = {"wall_cv": cv, "worst_relative_median_deviation": deviation,
                                "pass": cv <= 0.05 and deviation <= 0.10}
        summary[name] = {
            "pointwise_exact_count": int(proof[0]), "pointwise_nonzero_count": int(proof[1]),
            "result_min": float(proof[2]), "result_max": float(proof[3]),
            "per_arm_lookups_per_round": expected_calls,
            "retained_wall_ns_per_lookup": rates, "quality": quality,
            "observed_release_over_portable_median_wall": statistics.median(rates["release"]) / statistics.median(rates["portable"]),
            "measurements": samples,
        }
    receipt(output / "summary.json", {
        "prepared": {"path": str(prepared), **identify(prepared / "sources.json")},
        "replay": {"path": str(binary), **identify(binary)},
        "release_binary_and_source_receipt": identify(binary.with_name("source-binaries.sha256")),
        "measurement": "Alternating paired rounds in a single loaded process per radius. First round discarded."
                       " The loop includes only coordinate iteration, getValue and checksum accumulation. Input"
                       " reads, ROOT and JSON loads, table construction, coordinate conversions, and harness allocations"
                       " precede timing. Any work or allocations inside either getValue are in the timed region.",
        "classification": "Kernel replay. Interpret ratios after resource and spread review.",
        "cases": summary,
    })
    require_cpu()


if __name__ == "__main__":
    APP()
