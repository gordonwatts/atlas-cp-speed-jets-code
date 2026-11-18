#!/usr/bin/env bash
set -eo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
name=${1:?new subdirectory name}
plan=$(realpath "${2:?frozen full-file paired plan}")
[[ "$name" =~ ^[a-zA-Z0-9_-]+$ ]]
out="$root/results/benchmarks/$name"
mkdir "$out"
exec > "$out/controller.log" 2>&1
trap 'echo $? > "$out/exit-code"' EXIT
source tools/reference/setup.sh
unset PYTHONPATH PYTHONHOME
sdk="$root/results/build/core-precision-002/venv/lib/python3.14/site-packages/correctionlib"
export LD_LIBRARY_PATH="$sdk/lib:$LD_LIBRARY_PATH"
py="$root/results/build/core-precision-002/venv/bin/python"
for scope in small large; do
  "$py" tools/count_resolution_draws.py "$JETS_REFERENCE_INPUT" "$out/count-$scope.npz" "$out/count-$scope.json" \
    "$root/results/build/rng-cache-counts-001/libjets.so" "$root/results/export/portable-001/jets.json" \
    "$root/results/export/portable-001/provenance.json" --scope "$scope" --events 20000
  "$py" - "$out/count-$scope.npz" "$root/results/reference/full-file-pilot-$scope-002/01-full/histograms.npz" <<'PY'
import sys
import numpy as np
with np.load(sys.argv[1],allow_pickle=False) as result, np.load(sys.argv[2],allow_pickle=False) as authority:
    if set(result.files) != set(authority.files) or not all(np.array_equal(result[k], authority[k]) for k in result.files):
        raise ValueError("Instrumented random-stream workflow differed from AnalysisBase")
PY
done
"$py" tools/time_columnar_trials.py "$plan" "$out/pairs"
