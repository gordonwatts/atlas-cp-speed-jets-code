#!/usr/bin/env bash
# Compare the original and buffer-reusing framework overlays on a fixed provider and document.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
test "$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/$$/status)" = 3
test "$#" -eq 7
output=$(realpath -m "$1")
scope=$2
case "$scope" in small|large|combined) ;; *) exit 2;; esac
off=$(realpath "$3")
on=$(realpath "$4")
provider=$(realpath "$5")
payload=$(realpath "$6")
sdk=$(realpath "$7")
mkdir "$output"
exec > "$output/controller.log" 2>&1
trap 'echo $? > "$output/exit-code"' EXIT
for name in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS BLIS_NUM_THREADS ROOT_MAX_THREADS; do
  export "$name=1"
done
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH
py="$root/.reference-venv/bin/python"
stock="$root/results/reference/full-file-pilot-$scope-002"
for switch in off on; do
  if [[ $switch == off ]]; then overlay=$off; else overlay=$on; fi
  date -Is
  echo "$switch"
  pilot="$output/$switch"
  if "$py" framework/bench_path_b.py "$pilot" "$overlay" "$provider" "$payload" "$sdk" --scope "$scope" --stock "$stock" --events 20000 --pilot; then
    echo 0 > "$output/$switch-pilot.exit"
  else
    echo "$?" > "$output/$switch-pilot.exit"
  fi
  "$py" tools/benchmark_campaign.py recover-b-pilot "$pilot" "$output/$switch-physics"
done
