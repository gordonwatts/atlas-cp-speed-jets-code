#!/usr/bin/env bash
# Build and runtime stages are intentionally independent CPU allocations.
# Usage: tools/qualify_rng_prefix.sh NEW-TAG build [screen-events]
#        tools/qualify_rng_prefix.sh TAG screen|confirm
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
tag=${1:?fresh result label}
stage=${2:?build, screen or confirm}
[[ $tag =~ ^[a-z0-9-]+$ ]]
case "$stage" in
  build)
    screen_events=${3:-32}
    [[ $# -le 3 && $screen_events =~ ^[1-9][0-9]{0,4}$ ]]
    (( screen_events <= 20000 ))
    ;;
  screen|confirm) [[ $# -eq 2 ]] ;;
  *) echo "Select build, screen or confirm" >&2; exit 2 ;;
esac

out="$root/results/diagnostics/rng-prefix-$tag"
bundle="$root/provenance/provider-patches/rng-prefix"
py="$root/.reference-venv/bin/python"
gcc_lib=/cvmfs/sft.cern.ch/lcg/releases/gcc/15.2.0-35657/x86_64-el9/lib64
for key in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS \
           VECLIB_MAXIMUM_THREADS BLIS_NUM_THREADS ROOT_MAX_THREADS RAYON_NUM_THREADS; do
  export "$key=1"
done
unset PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1

if [[ $stage == build ]]; then
  # Refuse to apply, revert or synthesize active source as part of the run.
  git apply --reverse --check "$bundle/integration.diff"
  grep -Eq '^[[:space:]]*rng-prefix\).*\(-DJETS_RNG_PREFIX\)' tools/build-portable.sh
  mkdir "$out"
else
  test "$(cat "$out/build/exit-code")" = 0
  # Share the guest's analysis slot with the formal campaign driver.
  exec 9>>"$root/results/.benchmark-cpu3.lock"
  flock --exclusive --nonblock 9
fi
mkdir "$out/$stage"
exec > "$out/$stage/controller.log" 2>&1
trap 'printf "%s\n" "$?" > "$out/$stage/exit-code"' EXIT

fixtures="$out/build/fixtures"
if [[ $stage == build ]]; then
  for arm in off on; do
    options=(rng-cache-large fast-variable-flow)
    if [[ $arm == on ]]; then options+=(rng-prefix); fi
    taskset -c 5 env -u LD_LIBRARY_PATH -u LD_PRELOAD -u JETS_PORTABLE_BUILD_SNAPSHOT \
      bash tools/build-portable.sh "rng-prefix-$arm-$tag" core-precision-002 portable-lwtnn-001 "${options[@]}"
  done
  taskset -c 5 env -u LD_LIBRARY_PATH -u LD_PRELOAD bash "$bundle/build-fixture.sh" "$fixtures" 262144
  taskset -c 5 "$py" -m tools.prepare_rng_prefix_pair freeze "$tag" "$out" "$screen_events"
  exit 0
fi

test "$(cat "$fixtures/build-exit-code")" = 0
test "$(cat "$fixtures/cache-slots.txt")" = 262144
taskset -c 3 sha256sum --check "$fixtures/inputs.sha256"
taskset -c 3 sha256sum --check "$out/inputs.sha256"
taskset -c 3 "$py" -m tools.prepare_rng_prefix_pair check-observer "$out/screen.json"

if [[ $stage == screen ]]; then
  release_libs=$(cat "$fixtures/library-path.txt")
  for check in raw-and-gaussian resolution-baseline resolution-candidate; do
    taskset -c 3 env -u LD_PRELOAD "LD_LIBRARY_PATH=$release_libs" \
      "$fixtures/$check" > "$out/screen/$check.log" 2>&1
  done
  grep -Eq '^PASS known answers, zero raw words, prefix and full continuation,' "$out/screen/raw-and-gaussian.log"
  for arm in baseline candidate; do
    grep -Fxq 'cache_slots 262144' "$out/screen/resolution-$arm.log"
    grep -Eq '^bitwise ROOT checks [1-9][0-9]*$' "$out/screen/resolution-$arm.log"
  done
  cmp "$out/screen/resolution-baseline.log" "$out/screen/resolution-candidate.log"
  sha256sum "$out/screen"/{raw-and-gaussian,resolution-baseline,resolution-candidate}.log \
    > "$out/screen/fixture-logs.sha256"
  taskset -c 3 env -u LD_LIBRARY_PATH -u LD_PRELOAD LANG=C.UTF-8 \
    bash tools/screen_trials.sh "$out/screen.json" "$out/screen/pair"
  taskset -c 3 sha256sum --check "$out/inputs.sha256"
  taskset -c 3 "$py" -m tools.prepare_rng_prefix_pair check-observer "$out/screen.json"
  taskset -c 3 "$py" -m tools.prepare_rng_prefix_pair seal-screen "$out"
else
  test "$(cat "$out/screen/exit-code")" = 0
  taskset -c 3 sha256sum --check "$out/screen/fixture-logs.sha256"
  taskset -c 3 sha256sum --check "$out/screen-evidence.sha256"
  test -s "$gcc_lib/libstdc++.so.6"
  test -s "$gcc_lib/libgcc_s.so.1"
  taskset -c 3 env -u LD_PRELOAD "LD_LIBRARY_PATH=$gcc_lib" LANG=C.UTF-8 \
    "$py" tools/time_columnar_trials.py "$out/confirm.json" "$out/confirm/workflow"
  taskset -c 3 sha256sum --check "$out/inputs.sha256"
  taskset -c 3 "$py" -m tools.prepare_rng_prefix_pair check-confirmation "$out"
fi
