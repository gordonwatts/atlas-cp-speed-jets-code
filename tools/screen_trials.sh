#!/usr/bin/env bash
# Run a bounded independent check before the full-file one-change comparisons.
set -eo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
plan=$(realpath "${1:?supply the one-change plan}")
result=$(realpath -m "${2:?supply an unused screening directory}")
if [[ $(awk '/^Cpus_allowed_list:/ {print $2}' /proc/$$/status) != 3 ]]; then
  echo "Reserve CPU 3 for this sequential screen" >&2
  exit 2
fi
source tools/reference/setup.sh
unset PYTHONPATH PYTHONHOME
sdk="$root/results/build/core-precision-002/venv/lib/python3.14/site-packages/correctionlib"
export LD_LIBRARY_PATH="$sdk/lib:$LD_LIBRARY_PATH"
py="$root/results/build/core-precision-002/venv/bin/python"
mkdir "$result"
exec > "$result/controller.log" 2>&1
trap 'echo $? > "$result/exit-code"' EXIT
"$py" - "$root" "$plan" "$result" <<'PY'
import json
import subprocess
import sys
from pathlib import Path
root,plan,output=map(Path,sys.argv[1:])
config=json.loads(plan.read_text())
authority=root/'results/reference/full-clean-001/reference.root'
for name,entry in config['variants'].items():
    if name=='portable':
        continue
    scopes=entry['scopes']
    for scope in (['both'] if 'both' in scopes else scopes):
        result=output/f'objects-{name}-{scope}.json'
        command=[config['python'],str(root/'tools/compare_columnar.py'),config['input'],str(authority),str(result),
                 '--scope',scope,'--mode','full','--provider',entry['provider'],'--payload',entry['payloads'][scope],
                 '--provenance',config['provenance'],'--events','32','--chunk','8',*entry.get('options',[])]
        print(command,flush=True)
        with (output/f'objects-{name}-{scope}.log').open('x') as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        if not json.loads(result.read_text())['passed']:
            raise RuntimeError(f'Independent comparison failed for {name}/{scope}')
PY
"$py" tools/time_columnar_trials.py "$plan" "$result/workflow"
