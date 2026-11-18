#!/usr/bin/env bash
# Run the nominal checkpoint through the same CPRun entry point and observer as A.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
overlay=$(realpath "${1:?supply the runtime overlay library directory}")
provider=$(realpath "${2:?supply the shared jet node provider}")
payload=$(realpath "${3:?supply the shared jet payload}")
result=$(realpath -m "${4:?supply a fresh framework result directory}")
events=${5:?supply the event count}
jets_framework_mode=${6:-nominal}
sdk=$(realpath "${JETS_CORRECTIONLIB_SDK:-$root/results/build/core-001/venv/lib/python3.14/site-packages/correctionlib}")
core="$sdk/lib/libcorrectionlib.so"
if [[ "$jets_framework_mode" != nominal && "$jets_framework_mode" != full ]]; then
  echo "Framework mode must be nominal or full" >&2
  exit 2
fi
test -s "$provider"
test -s "$payload"
test -s "$core"
test -s "$overlay/libJetCalibToolsLib.so"
test -s "$overlay/libJetsCorrectionRuntime.so"
mkdir "$result"
cd "$result"
trap 'echo $? > exit-code' EXIT

set +eu
source "$root/tools/reference/setup.sh" > setup.log 2>&1
setup_status=$?
set -eu
test "$setup_status" -eq 0
export PYTHONPATH="$root/tools/reference:$PYTHONPATH"
export LD_LIBRARY_PATH="$overlay:$sdk/lib:$LD_LIBRARY_PATH"
export JETS_PROVIDER="$provider" JETS_PAYLOAD="$payload"
export JETS_FRAMEWORK_TRACE="$result/framework-trace.txt"
{
  date --iso-8601=seconds
  grep Cpus_allowed_list /proc/self/status
  g++ --version
  root-config --version
  command -v CPRun.py
} > environment.txt
sha256sum "$overlay"/*.so "$core" "$provider" "$payload" "$root/config/jets-observer.yaml" > runtime-inputs.sha256
stat -Lc '%s %n' "$JETS_REFERENCE_INPUT" > input.stat
args=(-i "$JETS_REFERENCE_INPUT" -t "$root/config/jets-observer.yaml" -o framework.root -e "$events"
      --direct-driver --merge-output-files)
if [[ "$jets_framework_mode" == nominal ]]; then args+=(--no-systematics); fi
{
  printf 'CPRun.py '
  printf '%q ' "${args[@]}"
  printf '\n'
} > command.txt
set +e
TIMEFORMAT='external_wall_s=%R user_s=%U sys_s=%S'
(time CPRun.py "${args[@]}" > job.log 2>&1) 2> framework.time
status=$?
set -e
printf '%d\n' "$status" > job-exit-code
test "$status" -eq 0
grep -q 'worker finished successfully' job.log
test -s framework.root
grep -Fqx "loaded $overlay/libJetCalibToolsLib.so" framework-trace.txt
grep -Fqx "loaded $overlay/libJetsCorrectionRuntime.so" framework-trace.txt
grep -Fqx "loaded $provider" framework-trace.txt
grep -Fqx "loaded $core" framework-trace.txt
for stage in small_pileup small_eta_jes small_gsc large_eta_jes large_jms; do
  grep -Eq "^count $stage [1-9][0-9]*$" framework-trace.txt
done
