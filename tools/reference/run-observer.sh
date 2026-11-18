#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
result=$(realpath -m "$1")
events=$2
mkdir "$result"
(
  set +u
  source "$root/tools/reference/setup.sh" > "$result/setup.log" 2>&1
  set -u
  export PYTHONPATH="$root/tools/reference:$PYTHONPATH"
  cd "$result"
  date --iso-8601=seconds > started.txt
  stat -Lc '%s %n' "$JETS_REFERENCE_INPUT" > input.stat
  sha256sum "$JETS_REFERENCE_INPUT" > input.sha256
  {
    uname -a
    lscpu
    free -h
    python -V
    g++ --version
    root-config --version
    git -C "$root" rev-parse HEAD
  } > environment.txt
  args=(-i "$JETS_REFERENCE_INPUT" -t "$root/config/jets-observer.yaml"
        -o reference.root -e "$events" --direct-driver --merge-output-files
        --run-perf-stat --algorithm-timers)
  printf 'taskset -c 3 CPRun.py ' > command.txt
  printf '%q ' "${args[@]}" >> command.txt
  printf '\n' >> command.txt
  set +e
  TIMEFORMAT='external_wall_s=%R user_s=%U sys_s=%S'
  (time taskset -c 3 CPRun.py "${args[@]}" > job.log 2>&1) 2> observer.time
  status=$?
  set -e
  printf '%s\n' "$status" > exit-code.txt
  date --iso-8601=seconds > ended.txt
  test "$status" -eq 0
  grep -q 'worker finished successfully' job.log
  test -s reference.root
)
input=$(cut -d ' ' -f3- "$result/input.sha256")
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
taskset -c 3 "$root/.reference-venv/bin/python" "$root/tools/reference/verify_observer.py" \
  "$result/reference.root" "$input" "$root/config/jet-states.json" \
  "$result/initialized-tools.tsv" "$result/verification.json" "$events"
