#!/usr/bin/env bash
# Short controlled diagnostics. Production retained timings use benchmark_campaign.py.
set -eo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
out="$root/results/diagnostics/${1:?fresh comparison tag required}"
mkdir "$out"
cd "$root"
exec > "$out/run.log" 2>&1
trap 'echo $? > "$out/exit-code"' EXIT
source tools/reference/setup.sh
unset PYTHONHOME PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 ROOT_MAX_THREADS=1
base="$root/results/build/core-precision-002/venv"
sdk="$base/lib/python3.14/site-packages/correctionlib"
export LD_LIBRARY_PATH="$sdk/lib:$LD_LIBRARY_PATH"
for scope in small large both; do
  for variant in faithful portable; do
    if [[ "$variant" == faithful ]]; then
      provider="$root/results/build/complete-005/libjets.so"
      payload="$root/results/export/complete-005/payload/jets.json"
    else
      provider="$root/results/build/portable-003/libjets.so"
      payload="$root/results/export/portable-001/jets.json"
    fi
    export JETS_PHASE_OUTPUT="$out/$variant-$scope.json"
    taskset -c 3 "$base/bin/python" tools/profile_columnar.py "$JETS_REFERENCE_INPUT" "$out/$variant-$scope.npz" \
      --provider "$provider" --payload "$payload" --mode full --scope "$scope" -e 2000 --chunk 1000
  done
done
"$base/bin/python" - "$out" <<'PY'
import json,numpy as np,sys
from pathlib import Path
dest=Path(sys.argv[1])
for scope in ('small','large','both'):
 with np.load(dest/f'faithful-{scope}.npz') as a,np.load(dest/f'portable-{scope}.npz') as c:
  assert a.files==c.files and all(np.array_equal(a[k],c[k]) for k in a.files)
PY
