#!/usr/bin/env bash
# Run one frozen shipment qualification through the existing acceptance tools.
# Usage: tools/verify_shipment_workflows.sh SPEC NEW_CONTROLLER_DIR
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
[[ $# -eq 2 ]] || { echo "Usage: $0 SPEC NEW_CONTROLLER_DIR" >&2; exit 2; }
spec=$(realpath -e "$1")
controller=$(realpath -m "$2")
py="$root/.reference-venv/bin/python"
mkdir -p "$(dirname "$controller")"
mkdir "$controller"
frozen="$controller/specification.json"
cp -- "$spec" "$frozen"
sha256sum -- "$frozen" > "$controller/specification.sha256"
sha256sum -- "$root/tools/verify_shipment_workflows.sh" > "$controller/controller.sha256"
for name in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS \
            VECLIB_MAXIMUM_THREADS BLIS_NUM_THREADS ROOT_MAX_THREADS RAYON_NUM_THREADS; do
  export "$name=1"
done
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD JETS_PROVIDER JETS_PAYLOAD \
      ATLAS_JET_LIBRARY_AUDIT JETS_FRAMEWORK_TRACE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1

active=
timestamp() { date -u +'%Y-%m-%dT%H:%M:%S.%NZ'; }
finish() {
  local status=$?
  if [[ -n "$active" && -d "$active" && ! -e "$active/exit-code" ]]; then
    timestamp > "$active/ended.utc"
    printf '%s\n' "$status" > "$active/exit-code"
    printf '%s\t%s\t%s\n' "$(timestamp)" "$active" "$status" >> "$controller/stages.tsv"
  fi
  timestamp > "$controller/ended.utc"
  printf '%s\n' "$status" > "$controller/exit-code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
timestamp > "$controller/started.utc"
printf 'utc\tstage\tstatus\n' > "$controller/stages.tsv"

run_stage() {
  local name=$1 cpu=$2 status=0
  shift 2
  active="$controller/$name"
  mkdir "$active"
  timestamp > "$active/started.utc"
  printf '%s\n' "$cpu" > "$active/cpu"
  printf '%q ' taskset -c "$cpu" "$@" > "$active/command.txt"
  printf '\n' >> "$active/command.txt"
  printf '%s\t%s\tstarted\n' "$(timestamp)" "$active" >> "$controller/stages.tsv"
  printf '%s CPU %s\n' "$name" "$cpu"
  {
    sha256sum --check --quiet "$controller/specification.sha256" && taskset -c "$cpu" "$@"
  } > "$active/stage.log" 2>&1 || status=$?
  timestamp > "$active/ended.utc"
  printf '%s\n' "$status" > "$active/exit-code"
  printf '%s\t%s\t%s\n' "$(timestamp)" "$active" "$status" >> "$controller/stages.tsv"
  if (( status != 0 )); then
    printf 'Qualification stopped at %s (exit %s). See %s/stage.log\n' "$name" "$status" "$active" >&2
  fi
  active=
  return "$status"
}

run_stage 00-preflight 5 "$py" - "$root" "$controller" "$spec" "$frozen" <<'PY'
from pathlib import Path
from hashlib import file_digest
import json
import os
import sys

root, controller, spec, frozen = map(Path, sys.argv[1:])

def read(path):
    return json.loads(path.read_text())

def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()

def member(directory, name):
    if Path(name).name != name:
        raise ValueError(f"Unsafe bundle member: {name}")
    return (directory / name).resolve(strict=True)

def verified(path, identity):
    if path.stat().st_size != identity["bytes"] or sha(path) != identity["sha256"]:
        raise ValueError(f"Frozen artifact differs: {path}")

if spec.read_bytes() != frozen.read_bytes():
    raise ValueError("Qualification specification changed while it was copied")
plan = read(frozen)
repeats = plan.get("profile_repeats", 10)
extra = plan.get("unprofiled_overhead", False)
if type(repeats) is not int or repeats < 2 or type(extra) is not bool or (extra and repeats != 10):
    raise ValueError("Use at least two profile passes. The optional unprofiled overhead diagnostic requires ten.")
paths = {key: (root / plan[key]).resolve(strict=True) for key in
         ("overlay", "provider", "payload", "sdk", "bundle", "installation", "partition")}
output = (root / plan["output"]).resolve()
output_names = (
    "native-profile", "native-profile-decoded", "phases", "chunk-order",
    "C-both", "B-combined", "shipped-both", "shipped-small", "shipped-large",
    "C-small", "C-large", "B-small", "B-large", "recovered-small", "recovered-large",
    "B-small-pilot.exit", "B-large-pilot.exit",
)
if extra:
    output_names += ("native-unprofiled",)
if output == controller or output.is_relative_to(controller):
    raise ValueError("Keep the qualification artifact root outside the stage controller directory")
for name in output_names:
    path = output / name
    if controller.is_relative_to(path) or path.exists() or path.is_symlink():
        raise ValueError(f"Qualification needs a new stage artifact path: {path}")
frozen_input = root / "config/jets-input.json"
source = read(frozen_input)
original = Path(source["path"]).resolve(strict=True)
verified(original, source)
if source["entry_interval"] != [0, 20000]:
    raise ValueError("The final qualification is defined for the original full file")

bundle = paths["bundle"]
bundle_manifest = bundle / "bundle.json"
manifest = read(bundle_manifest)
wheel = member(bundle, manifest["wheel"]["name"])
demo = member(bundle, manifest["visible_script"]["name"])
for path, identity in ((wheel, manifest["wheel"]), (demo, manifest["visible_script"]),
                       (paths["provider"], manifest["provider"]), (paths["payload"], manifest["payload"])):
    verified(path, identity)
defaults = manifest["demo_defaults"]
if (defaults["scope"], defaults["mode"]) != ("both", "full"):
    raise ValueError("The final visible workflow must default to combined full calibration")
install_manifest = paths["installation"] / "installations.json"
installed = read(install_manifest)
if (installed["wheel_sha256"] != manifest["wheel"]["sha256"]
        or installed["script_sha256"] != manifest["visible_script"]["sha256"]
        or installed["demo_defaults"] != defaults):
    raise ValueError("The installation belongs to a different shipment")
# Passing the venv executable's symlink path is required by CPython.
installed_python = paths["installation"] / "pip/env/bin/python"
if not os.access(installed_python, os.X_OK):
    raise ValueError(f"Missing installed interpreter: {installed_python}")

reference = root / "results/reference"
observer = (reference / "full-clean-001/reference.root").resolve(strict=True)
observation = observer.with_name("verification.json")
verification = read(observation)
if verification["events"] != 20000 or verification["source_file_uuid"] != source["tfile_uuid"]:
    raise ValueError("Stock observer does not cover this full original file")
stocks = {scope: reference / f"full-file-pilot-{suffix}-002" for scope, suffix in
          (("both", "combined"), ("small", "small"), ("large", "large"))}
short = reference / "benchmark-pilot-combined-002"
for stock_dir, scope, events in [(stocks["both"], "combined", 20000), (stocks["small"], "small", 20000),
                                 (stocks["large"], "large", 20000), (short, "combined", 32)]:
    provenance = read(stock_dir / "provenance.json")
    if (provenance.get("path", "A") != "A" or provenance["events"] != events
            or provenance["scope_collections"] != scope or provenance["input_sha256"] != source["sha256"]
            or Path(provenance["input"]).resolve(strict=True) != original):
        raise ValueError(f"Stock histogram provenance differs: {stock_dir}")
    for leg in ("framework", "histogram"):
        process = read(stock_dir / "01-full" / f"{leg}.json")
        if process["exit"] or process["threads_seen"] != [1] or process["cpus_seen"] != ["3"]:
            raise ValueError(f"Unqualified stock receipt: {stock_dir} {leg}")
stock_histograms = {scope: (stock / "01-full/histograms.npz").resolve(strict=True) for scope, stock in stocks.items()}
stock_32 = (short / "01-full/histograms.npz").resolve(strict=True)
if not output.is_dir():
    output.mkdir(parents=True)
with (controller / "resolved.json").open("x") as stream:
    json.dump({
        "source_specification": str(spec), "specification_sha256": sha(frozen), "output": str(output),
        "original": {**source, "resolved_path": str(original)}, "artifact_paths": {k: str(v) for k, v in paths.items()},
        "bundle_manifest_sha256": sha(bundle_manifest), "installation_receipt_sha256": sha(install_manifest),
        "installed_python": str(installed_python), "demo": str(demo), "wheel": str(wheel),
        "observer": str(observer), "observer_receipt_sha256": sha(observation), "stock_directories": {
            scope: str(path) for scope, path in stocks.items()
        }, "stock_histograms": {scope: {"path": str(path), "sha256": sha(path)}
                                for scope, path in {**stock_histograms, "both_32": stock_32}.items()},
        "profile_repeats": repeats, "unprofiled_overhead": extra, "calibration_cpu": 3,
        "analysis_threads": 1, "independent_offline_cpu": 5,
    }, stream, indent=2, allow_nan=False)
    stream.write("\n")
fields = (output, original, installed_python, demo, wheel, paths["installation"], observer,
          stocks["both"], stocks["small"], stocks["large"], stock_32, int(extra))
with (controller / "paths.nul").open("xb") as stream:
    for value in fields:
        stream.write(os.fsencode(str(value)) + b"\0")
PY

mapfile -d '' -t fields < "$controller/paths.nul"
[[ ${#fields[@]} == 12 ]]
output=${fields[0]}
original=${fields[1]}
installed_python=${fields[2]}
demo=${fields[3]}
wheel=${fields[4]}
installation=${fields[5]}
observer=${fields[6]}
stock_both=${fields[7]}
stock_small=${fields[8]}
stock_large=${fields[9]}
stock_32=${fields[10]}
extra=${fields[11]}
qualify=("$py" tools/qualify_shipment.py "$frozen")

run_stage 01-native-record 3 "${qualify[@]}" record-profile --scope both
run_stage 02-native-decode 5 "$py" tools/native_stack_profile.py "$output/native-profile" \
  --analysis-output "$output/native-profile-decoded"
if [[ $extra == 1 ]]; then
  run_stage 02b-native-unprofiled 3 "$py" - "$root" "$controller" "$output" <<'PY_AUDIT'
from pathlib import Path
import json
import re
import sys

root, controller, output = map(Path, sys.argv[1:])
sys.path.insert(0, str(root / "tools"))
sys.path.insert(0, str(root / "tools/reference"))
from benchmark import run_leg
from profile_final_demo import histogram_comparison
from profile_native import PASS, digest, verify_loaded, write_json

def read(path):
    return json.loads(path.read_text())

settings = read(controller / "resolved.json")
native = output / "native-profile"
record = read(native / "recording.json")
perf_argv = read(native / "command.json")
profiled_argv = perf_argv[perf_argv.index("--") + 1:]
expected = [settings["installed_python"], settings["demo"], settings["original"]["resolved_path"],
            str(native / "histogram.npz"), "-n", "10"]
if (profiled_argv != expected or not record["passed"] or record["repeat"] != 10
        or record["scope"] != "both" or record["events_per_pass"] != 20000):
    raise ValueError("Unprofiled check requires the exact recorded installed ten-pass combined command")
stock = settings["stock_histograms"]["both"]
stock_histogram = Path(stock["path"])
profiled_histogram = Path(record["histogram"]["path"])
if digest(stock_histogram) != stock["sha256"] or digest(profiled_histogram) != record["histogram"]["sha256"]:
    raise ValueError("Stock or profiled histogram changed")

directory = output / "native-unprofiled"
directory.mkdir()
histogram = directory / "histogram.npz"
argv = profiled_argv.copy()
argv[3] = str(histogram)
write_json(directory / "command.json", {"recorded_visible_argv": profiled_argv, "unprofiled_argv": argv})
env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
       "PYTHONDONTWRITEBYTECODE": "1", "ATLAS_JET_LIBRARY_AUDIT": str(directory / "library-maps.json")}
env.update(dict.fromkeys(("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                         "BLIS_NUM_THREADS", "ROOT_MAX_THREADS", "RAYON_NUM_THREADS"), "1"))
original = Path(settings["original"]["resolved_path"])
process = run_leg(argv, env, directory, "application", event_source=original)
if not process["source_mapping_observed"]:
    raise ValueError("The original PHYSLITE was not mapped by the unprofiled worker")
log = (directory / "application.log").read_text()
visits = [match.groups() for line in log.splitlines() if (match := PASS.fullmatch(line))]
markers = [(str(row["pass"]), str(row["of"]), str(row["events"]), row["objects"]) for row in record["pass_ends"]]
branch_pattern = r"(?m)^Input branch SHA-256: ([0-9a-f]{64})$"
branches = re.findall(branch_pattern, log)
recorded_branches = re.findall(branch_pattern, (native / "application.log").read_text())
if (visits != markers or len(visits) != 10 or len(branches) != 1 or branches != recorded_branches
        or f"({settings['original']['tfile_uuid']})" not in log
        or log.count("20000 unique events, 200000 event visits across 10 passes") != 1):
    raise ValueError("Unprofiled command differs in passes, physical branches, source identity or jet counts")
audited = verify_loaded(directory, record["wheel"])
stock_fields = histogram_comparison(stock_histogram, histogram, "both", 20000)
profiled_fields = histogram_comparison(profiled_histogram, histogram, "both", 20000)
passed = all(stock_fields.values()) and all(profiled_fields.values())
write_json(directory / "diagnostic.json", {
    "passed": passed, "classification": "unprofiled diagnostic outside the formal timing campaign",
    "profile_recording": str(native / "recording.json"), "repeat": 10, "events_per_pass": 20000,
    "input_branch_sha256": branches[0], "stock_exact_fields": stock_fields, "profiled_exact_fields": profiled_fields,
    "histogram_sha256": digest(histogram), "wall_s": process["wall_s"], "process_receipt": str(directory / "application.json"),
    "physically_mapped_libraries": audited,
})
if not passed:
    raise ValueError("Unprofiled diagnostic differs from the saved stock or profiled histograms")
PY_AUDIT
fi
run_stage 03-final-phases 3 "$py" tools/profile_final_demo.py \
  "$installed_python" "$demo" "$wheel" "$original" "$output/phases" \
  --stock-both "$stock_both" --stock-small "$stock_small" --stock-large "$stock_large"
run_stage 04-chunk-order 3 "$py" packaging/check_chunk_order.py check "$installation" \
  "$original" "$observer" "$stock_32" "$output/chunk-order"
run_stage 05-combined-objects 3 "${qualify[@]}" objects --scope both
run_stage 06-combined-framework 3 "${qualify[@]}" framework --scope both
run_stage 07-shipped-combined 3 "${qualify[@]}" shipped --scope both
run_stage 08-shipped-small 3 "${qualify[@]}" shipped --scope small
run_stage 09-shipped-large 3 "${qualify[@]}" shipped --scope large
run_stage 10-small-objects 3 "${qualify[@]}" objects --scope small
run_stage 11-large-objects 3 "${qualify[@]}" objects --scope large
run_stage 12-small-framework 3 "${qualify[@]}" framework --scope small
run_stage 13-large-framework 3 "${qualify[@]}" framework --scope large
