#!/usr/bin/env bash
# Run the combined full-file observer and independent histogram acceptance.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
overlay=$(realpath "${1:?supply the runtime overlay library directory}")
provider=$(realpath "${2:?supply the shared provider}")
payload=$(realpath "${3:?supply the shared payload}")
export JETS_CORRECTIONLIB_SDK=$(realpath "${4:?supply the shared core SDK}")
result=$(realpath -m "${5:?supply a fresh result directory}")
reference=$(realpath "${6:?supply the stock full-file observer ROOT file}")
stock_histogram=$(realpath "${7:?supply the stock full-file histogram archive}")
events=20000
if [[ $(awk '/^Cpus_allowed_list:/ {print $2}' /proc/$$/status) != 3 ||
      $(awk '/^Threads:/ {print $2}' /proc/$$/status) != 1 ]]; then
  echo "Run the observer controller on CPU 3 with one thread" >&2
  exit 2
fi
for name in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS \
            VECLIB_MAXIMUM_THREADS BLIS_NUM_THREADS ROOT_MAX_THREADS; do
  export "$name=1"
done

./tools/run-framework.sh "$overlay" "$provider" "$payload" "$result" "$events" full &
job=$!
python3 framework/sample_processes.py "$job" "$result/process-samples.jsonl" &
sampler=$!
set +e
wait "$job"
job_status=$?
wait "$sampler"
sampler_status=$?
set -e
test "$job_status" -eq 0
test "$sampler_status" -eq 0

python3 - "$result" <<'PY'
import json
from pathlib import Path
import sys

result = Path(sys.argv[1])
jobs = []
for line in (result / "process-samples.jsonl").read_text().splitlines():
    for row in json.loads(line)["processes"]:
        command = row["exec"].split()
        if (len(command) > 1 and Path(command[0]).name.startswith("python")
                and Path(command[1]).name == "CPRun.py"):
            jobs.append(row)
pids = sorted({job["pid"] for job in jobs})
states = sorted({(job["threads"], job["allowed_cpus"]) for job in jobs})
passed = len(pids) == 1 and states == [(1, "3")]
with (result / "process-verification.json").open("x") as stream:
    json.dump({"passed": passed, "processes": pids, "samples": len(jobs), "states": states}, stream, indent=2)
    stream.write("\n")
if not passed:
    raise SystemExit("CPRun process sampling did not verify one thread on CPU 3")
PY

python="$root/.reference-venv/bin/python"
"$python" tools/reference/workflow.py histogram "$result/framework.root" config/jet-states.json \
  config/jet-histogram.json "$result/histograms.npz" full --scope combined
"$python" framework/compare_observer.py "$reference" "$result/framework.root" "$result/observer-comparison.json" \
  --require-full --stock-histogram "$stock_histogram" --framework-histogram "$result/histograms.npz" \
  > "$result/observer-comparison.log" 2>&1

input_record=$(cat "$result/input.stat")
input="${input_record#* }"
"$python" tools/reference/verify_observer.py "$result/framework.root" "$input" config/jet-states.json \
  "$result/initialized-tools.tsv" "$result/reference-coverage.json" "$events" \
  > "$result/reference-coverage.log" 2>&1
counter_path=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["reference_counters"])' \
  "$result/observer-comparison.json")
authority=("$reference")
if [[ "$counter_path" != "$reference" ]]; then authority+=("$counter_path"); fi
sha256sum "${authority[@]}" "$stock_histogram" "$result/framework.root" "$result/histograms.npz" \
  "$result/observer-comparison.json" "$result/reference-coverage.json" "$result/process-verification.json" \
  > "$result/acceptance.sha256"
