#!/usr/bin/env bash
# The source checkout must already contain the isolated variable-axis index patch.
# Usage: tools/qualify_axis_trial.sh NEW-TAG [all|build|screen|confirm] [screen-events]
set -eo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
tag=${1:?fresh label suffix}
stage=${2:-all}
screen_events=${3:-32}
[[ "$tag" =~ ^[a-z0-9-]+$ && "$screen_events" =~ ^[1-9][0-9]*$ ]]
(( screen_events <= 20000 ))
case "$stage" in
  all)
    bash "$root/tools/qualify_axis_trial.sh" "$tag" build "$screen_events"
    exec bash "$root/tools/qualify_axis_trial.sh" "$tag" screen
    ;;
  build|screen|confirm) ;;
  *) echo "Select all, build, screen or confirm" >&2; exit 2 ;;
esac

out="$root/results/diagnostics/axis-index-$tag"
bundle=provenance/provider-patches/variable-axis-index
payload="$root/results/export/eta-jes-packed-001/jets.json"
for limit in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS \
             VECLIB_MAXIMUM_THREADS BLIS_NUM_THREADS ROOT_MAX_THREADS RAYON_NUM_THREADS; do
  export "$limit=1"
done
unset PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1

if [[ "$stage" == build ]]; then
  grep -q 'variable-axis-index)' tools/build-portable.sh
  grep -q 'JETS_VARIABLE_AXIS_INDEX' portable/histogram.cxx
  mkdir "$out"
fi
mkdir "$out/$stage"
exec > "$out/$stage/controller.log" 2>&1
trap 'echo $? > "$out/$stage/exit-code"' EXIT

if [[ "$stage" == build ]]; then
  core=core-precision-002
  lwtnn=portable-lwtnn-001
  for variant in off on; do
    options=(rng-cache fast-variable-flow)
    if [[ "$variant" == on ]]; then options+=(variable-axis-index); fi
    taskset -c 5 bash tools/build-portable.sh "axis-index-$variant-$tag" "$core" "$lwtnn" "${options[@]}"
  done
  taskset -c 5 bash "$bundle/build-fixture.sh" "$out/build/fixtures" "$payload"
  taskset -c 5 .reference-venv/bin/python - "$tag" "$out" "$payload" "$screen_events" <<'PY'
from pathlib import Path
import sys

from tools.joint_trials import BUILD, REFERENCE, ROOT, SCOPES, checksums, emit, json_file, sha, verify_payload

tag, target, packed, events = sys.argv[1:]
out, payload = map(Path, (target, packed))
sdk = BUILD / "core-precision-002/venv/lib/python3.14/site-packages/correctionlib"
python = BUILD / "core-precision-002/venv/bin/python"
core = sdk / "lib/libcorrectionlib.so"
provenance = payload.with_name("provenance.json")
packing = json_file(payload.with_name("packing.json"))
partition = json_file(payload.parent.parent / "eta-jes-partition-001/manifest.json")
verify_payload(payload, packing["packed"])
verify_payload(payload, partition["canonical"])
if (json_file(provenance)["sha256"] != packing["packed"]["sha256"]
        or sha(provenance) != partition["provenance"]["sha256"]):
    raise ValueError("Packed payload and archived provenance differ")
if not python.is_file():
    raise FileNotFoundError(python)
variants = {}
source_snapshots = {}
revisions = set()
for arm in ("off", "on"):
    directory = BUILD / f"axis-index-{arm}-{tag}"
    required = {"rng-cache", "fast-variable-flow"}
    if arm == "on":
        required.add("variable-axis-index")
    if ((directory / "exit-code").read_text().strip() != "0"
            or set((directory / "build-choices.txt").read_text().splitlines()) != required
            or (directory / "rng-cache-slots.txt").read_text().strip() != "65536"):
        raise ValueError(f"Unexpected build options for {directory}")
    provider = directory / "libjets.so"
    identity = checksums(directory / "native.sha256")
    for path in (provider, core):
        if identity.get(path) != sha(path):
            raise ValueError(f"Unexpected binary bytes for {path}")
    source_snapshots[arm] = checksums(directory / "source-inputs.sha256")
    revisions.add((directory / "source-revision.txt").read_text().strip())
    variants[arm] = {
        "python": str(python), "sdk": str(sdk), "provider": str(provider),
        "payloads": dict.fromkeys(SCOPES, str(payload)),
        "provenances": dict.fromkeys(SCOPES, str(provenance)),
        "options": ["--reuse-nominal-nnjvt-update"],
    }
if len(revisions) != 1 or source_snapshots["off"] != source_snapshots["on"]:
    raise ValueError("Both axis providers must come from one source snapshot")

source = json_file(ROOT / "config/jets-input.json")
input_file = Path(source["path"])
verify_payload(input_file, source)
observed = REFERENCE / "full-clean-001"
stock = json_file(observed / "verification.json")
if stock["source_file_uuid"] != source["tfile_uuid"] or stock["events"] != 20000:
    raise ValueError("Stock observer does not cover the independent full input")
authority = {
    scope: REFERENCE / f"full-file-pilot-{prefix}-002/01-full/histograms.npz"
    for scope, prefix in zip(SCOPES, ("small", "large", "combined"), strict=True)
}
sources = ("columnar/jets.py", "columnar/inputs.py", "jet_transport.py", "config/jet-states.json",
           "config/jet-histogram.json", "tools/compare_columnar.py", "tools/time_columnar_trials.py",
           "tools/screen_trials.sh", "tools/qualify_axis_trial.sh")
plan = {
    "experiment": "variable-axis-index", "build_label": tag, "build_revision": revisions.pop(),
    "input": str(input_file), "input_identity": source,
    "scopes": list(SCOPES), "events": 20000, "screen_events": int(events),
    "script": str(ROOT / "columnar/jets.py"),
    "script_sources": {str(ROOT / name): sha(ROOT / name) for name in sources},
    "observer": str(observed / "reference.root"),
    "authority": {key: str(path.resolve(strict=True)) for key, path in authority.items()},
    "screen_each_scope": True, "require_matching_branches": True, "compiler_only_runtime": True,
    "variants": variants,
}
for name, rounds in (("screen", 1), ("confirm", 4)):
    emit(out / f"{name}.json", {**plan, "rounds": rounds})
inputs = (payload, provenance, core, *(Path(arm["provider"]) for arm in variants.values()),
          *authority.values(), out / "screen.json", out / "confirm.json")
with (out / "inputs.sha256").open("x") as receipt:
    for path in inputs:
        receipt.write(f"{sha(path)}  {path}\n")
PY
  exit 0
fi

test "$(cat "$out/build/exit-code")" = 0
taskset -c 3 sha256sum --check "$out/inputs.sha256"
fixtures="$out/build/fixtures"
test "$(cat "$fixtures/exit-code")" = 0
taskset -c 3 sha256sum --check "$fixtures/inputs.sha256"

if [[ "$stage" == screen ]]; then
  taskset -c 3 env -u LD_LIBRARY_PATH -u LD_PRELOAD LANG=C.UTF-8 \
    bash -e -o pipefail -s -- "$fixtures" "$out/screen" <<'BASH'
source tools/reference/setup.sh
fixtures=$1
receipts=$2
for arm in off on; do
  "$fixtures/axis-$arm" < "$fixtures/geometry.txt" > "$receipts/axis-$arm.log"
  grep -Eq '^Exact variable-bin comparisons [1-9][0-9]* across [1-9][0-9]* archived geometries and synthetic boundaries$' \
    "$receipts/axis-$arm.log"
  "$fixtures/oracle-$arm" > "$receipts/oracle-$arm.log"
  grep -Eq '^exact release comparisons: [1-9][0-9]*$' "$receipts/oracle-$arm.log"
done
cmp "$receipts/axis-off.log" "$receipts/axis-on.log"
cmp "$receipts/oracle-off.log" "$receipts/oracle-on.log"
sha256sum "$receipts"/{axis-off,axis-on,oracle-off,oracle-on}.log > "$receipts/fixture-logs.sha256"
BASH
else
  test "$(cat "$out/screen/exit-code")" = 0
  sha256sum --check "$out/screen/fixture-logs.sha256"
fi
taskset -c 3 bash tools/screen_trials.sh "$out/$stage.json" "$out/$stage/pair"
