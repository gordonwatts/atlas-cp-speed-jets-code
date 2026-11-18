#!/usr/bin/env bash
# Freeze one invocation and pair the existing production controller with guest telemetry.
set -euo pipefail
if [[ $# -lt 2 || $1 == --help ]]; then
  printf 'Usage: %s SHIPMENT.json NEW-TRACE-DIR [qualify_shipment time options]\n' "$0"
  if [[ ${1:-} == --help ]]; then exit 0; else exit 2; fi
fi
root=$(cd "$(dirname "$0")/.." && pwd)
specification=$(realpath "$1")
trace=$(realpath -m "$2")
shift 2
cd "$root"
test -s "$specification"
mkdir "$trace"
frozen="$trace/specification.json"
cp -- "$specification" "$frozen"
sha256sum -- "$frozen" > "$trace/specification.sha256"
sha256sum -- tools/run_production_with_monitor.sh tools/watch_guest.py tools/qualify_shipment.py \
  tools/benchmark_campaign.py tools/reference/benchmark.py > "$trace/controller.sha256"
date -u +'%Y-%m-%dT%H:%M:%S.%NZ' > "$trace/started.utc"
cat /proc/sys/kernel/random/boot_id > "$trace/boot-id"
df -P "$root" > "$trace/guest-disk.txt"
cpu_limits=(OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
            VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 ROOT_MAX_THREADS=1 RAYON_NUM_THREADS=1)
command=(python3 "$root/tools/watch_guest.py" "$trace/guest.jsonl" --
         taskset -c 3 env "${cpu_limits[@]}" "$root/.reference-venv/bin/python" -u
         "$root/tools/qualify_shipment.py" "$frozen" time "$@")
{
  printf '%q ' "${command[@]}"
  printf '\n'
} > "$trace/command.txt"
trap 'status=$?; date -u +"%Y-%m-%dT%H:%M:%S.%NZ" > "$trace/ended.utc"; printf "%d\n" "$status" > "$trace/exit-code"; exit "$status"' EXIT
sha256sum --check --quiet "$trace/specification.sha256"
"${command[@]}" > "$trace/controller.log" 2>&1
