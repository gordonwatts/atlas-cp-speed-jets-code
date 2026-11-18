#!/usr/bin/env bash
# Compare two independent implementations after their dedicated semantic fixtures.
set -eo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
suffix=${1:?fresh result suffix}
[[ "$suffix" =~ ^[a-z0-9-]+$ ]]
out="$root/results/diagnostics/structural-trials-$suffix"
mkdir "$out"
exec > "$out/controller.log" 2>&1
trap 'echo $? > "$out/exit-code"' EXIT
core=core-precision-002
sdk="$root/results/build/$core/venv/lib/python3.14/site-packages/correctionlib"
gcc=/cvmfs/sft.cern.ch/lcg/releases/gcc/15.2.0-35657/x86_64-el9/lib64
py="$root/results/build/$core/venv/bin/python"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ROOT_MAX_THREADS=1
unset PYTHONPATH PYTHONHOME LD_PRELOAD
run() {
  local name=$1
  shift
  echo "$(date -Is) $name" >> "$out/tasks.txt"
  if "$@" > "$out/$name.log" 2>&1; then
    echo "0 $name" >> "$out/status.txt"
    return 0
  else
    local code=$?
    echo "$code $name" >> "$out/status.txt"
    return "$code"
  fi
}
template=results/diagnostics/joint-plans-001/rng-size.json
packs=provenance/provider-patches
sign="$root/results/build/joint-sign-$suffix/libjets.so"
nn="$root/results/build/joint-nn-inputs-$suffix/libjets.so"
fixtures="$out/nn-fixture"
if run build-sign taskset -c 5 bash tools/build-portable.sh "joint-sign-$suffix" "$core" portable-lwtnn-001 rng-cache unsigned-sign-pairs; then
  if run fixture-sign env LD_LIBRARY_PATH="$sdk/lib:$gcc" taskset -c 3 "$py" -m tools.test_unsigned_sign_pairs "$sign" "$out/sign-fixture.json"; then
    run plan-sign taskset -c 5 .reference-venv/bin/python "$packs/unsigned-sign-pairs/prepare-plan.py" "$template" "$sign" "$out/sign-plan.json" 1
  fi
fi
if run build-nn taskset -c 5 bash tools/build-portable.sh "joint-nn-inputs-$suffix" "$core" portable-lwtnn-001 rng-cache nnjvt-inputs; then
  if run compile-nn-fixture taskset -c 5 bash "$packs/nnjvt-inputs/build-fixture.sh" "$fixtures" "$root/results/export/portable-001/jets.json"; then
    if run fixture-nn env LD_LIBRARY_PATH="$(cat "$fixtures/library-path.txt")" taskset -c 3 "$fixtures/nnjvt-inputs-fixture" "$fixtures/baseline.so" "$fixtures/candidate.so" "$fixtures/nnjvt-update.json" "$fixtures/cuts-and-network.json"; then
      run plan-nn taskset -c 5 .reference-venv/bin/python tools/prepare_resolution_pair.py "$template" "$nn" "$out/nn-plan.json" --mechanism nnjvt-inputs
    fi
  fi
fi
for arm in sign nn; do
  if [[ -s "$out/$arm-plan.json" ]]; then
    run "screen-$arm" python3 tools/watch_guest.py "$out/$arm-guest.jsonl" -- taskset -c 3 bash tools/screen_trials.sh "$out/$arm-plan.json" "results/benchmarks/joint-$arm-structural-$suffix" || true
  fi
done
if awk '$1!=0 {bad=1} END {exit !bad}' "$out/status.txt"; then exit 1; fi
