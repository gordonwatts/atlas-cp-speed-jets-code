#!/usr/bin/env bash
set -eo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
original_ref=${1:?pass a full absolute oracle file}
provider=${2:?pass a full absolute provider}
payload=${3:?pass a full absolute payload}
label=${4:?new result directory name}
core=${5:-core-001}
mkdir -p "$root/results/checks"
mkdir "$root/results/checks/$label"
cd "$root/results/checks/$label"
exec > run.log 2>&1
trap 'echo $? > exit-code' EXIT
source "$root/tools/reference/setup.sh"
# Use release native libraries for the faithful checkpoint, with the standalone
# 3.14 Python and its own site-packages; keep all release Python modules out.
unset PYTHONPATH PYTHONHOME
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
sdk="$root/results/build/$core/venv/lib/python3.14/site-packages/correctionlib"
export LD_LIBRARY_PATH="$sdk/lib:$LD_LIBRARY_PATH"
taskset -c 3 "$root/results/build/$core/venv/bin/python" "$root/tools/compare_stages.py" \
  "$JETS_REFERENCE_INPUT" "$original_ref" "$provider" "$payload" "$PWD/agreement.json"
