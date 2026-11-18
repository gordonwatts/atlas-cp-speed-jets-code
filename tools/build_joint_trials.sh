#!/usr/bin/env bash
# Run sequentially in a build window, using new directories for each configuration.
set -eo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
trial=${1:?new label suffix}
[[ "$trial" =~ ^[a-z0-9-]+$ ]]
out="$root/results/build/joint-trials-$trial"
mkdir "$out"
exec > "$out/controller.log" 2>&1
trap 'echo $? > "$out/exit-code"' EXIT
git rev-parse HEAD > "$out/revision"
git status --porcelain --untracked-files=no > "$out/local-source-status"
py="$root/.reference-venv/bin/python"
sdk="$root/results/build/core-precision-002/venv/lib/python3.14/site-packages/correctionlib"
lw=portable-lwtnn-001
core=core-precision-002
candidate_core="core-plugin-batch-$trial"
new_sdk="$root/results/build/$candidate_core/venv/lib/python3.14/site-packages/correctionlib"
run() {
  local label=$1
  shift
  printf '%s %s\n' "$(date -Is)" "$label" >> "$out/tasks.txt"
  printf '%q ' "$@" >> "$out/commands.txt"
  printf '\n' >> "$out/commands.txt"
  if "$@" > "$out/$label.log" 2>&1; then
    printf '0 %s\n' "$label" >> "$out/status.txt"
  else
    local rc=$?
    printf '%s %s\n' "$rc" "$label" >> "$out/status.txt"
  fi
}
run pack taskset -c 5 "$py" -m exporters.pack_eta_jes results/export/portable-001/jets.json "results/export/eta-jes-packed-$trial"
if [[ -s "results/export/eta-jes-packed-$trial/jets.json" ]]; then
  run partition taskset -c 5 "$py" -m exporters.partition_payload "results/export/eta-jes-packed-$trial/jets.json" "results/export/eta-jes-partition-$trial"
  if [[ -s "results/export/eta-jes-partition-$trial/manifest.json" ]]; then
    run scope-index taskset -c 5 "$py" exporters/prepare_scope_index.py "results/export/eta-jes-packed-$trial/jets.json" "results/export/eta-jes-partition-$trial" "results/export/eta-jes-partition-$trial/payload-index.json"
  fi
fi
for variant in baseline bigger flow match; do
  options=(rng-cache)
  case "$variant" in
    bigger) options=(rng-cache-large) ;;
    flow) options+=(fast-variable-flow) ;;
    match) options+=(large-jmr-match-cache) ;;
  esac
  run "$variant" taskset -c 5 bash tools/build-portable.sh "joint-$variant-$trial" "$core" "$lw" "${options[@]}"
done
match="$root/results/build/joint-match-$trial/libjets.so"
if [[ -s "$match" ]]; then
  run focused-fixtures taskset -c 5 bash tools/build_joint_fixtures.sh "$out/fixtures" "$match" "$sdk"
fi
run generic-core taskset -c 5 bash tools/build-plugin-batch-trial.sh "$candidate_core"
run fixture-original taskset -c 5 bash tools/build-core-plugin-batch-fixture.sh "$sdk" "$out/original-sdk-fixture"
if [[ -s "$new_sdk/lib/libcorrectionlib.so" ]]; then
  run fixture-generic taskset -c 5 bash tools/build-core-plugin-batch-fixture.sh "$new_sdk" "$out/generic-sdk-fixture"
  run generic-provider taskset -c 5 bash tools/build-portable.sh "joint-generic-$trial" "$candidate_core" "$lw" rng-cache
fi
run framework-reuse taskset -c 4 bash tools/build-framework.sh "full-inputs-$trial" "$sdk" full reuse-inputs results/build/full-scoped-001
if awk '$1!=0{bad=1} END{exit !bad}' "$out/status.txt"; then exit 1; fi
