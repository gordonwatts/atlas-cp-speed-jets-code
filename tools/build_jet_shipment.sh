#!/usr/bin/env bash
# Build exactly one local wheel and install the visible example two independent ways.
set -eo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
serial=${1:?new positive integer wheel suffix}
core=${2:?evaluator build directory name}
provider=$(realpath "${3:?absolute or checkout-relative provider}")
payload=$(realpath "${4:?canonical payload}")
partition=$(realpath "${5:?audited partition}")
compact=${6:?off, small, large or both}
[[ "$serial" =~ ^[1-9][0-9]*$ && "$core" =~ ^[A-Za-z0-9-]+$ ]]
case "$compact" in off|small|large|both) ;; *) exit 2;; esac
shift 6
printf -v suffix '%03d' "$serial"
bundle="results/packaging/bundle-$suffix"
installation="results/packaging/pristine-$suffix"
test ! -e "$bundle"
test ! -e "$installation"
build=results/packaging/build-tools-001
gcc=/cvmfs/sft.cern.ch/lcg/releases/gcc/15.2.0-35657/x86_64-el9/lib64
uv=/cvmfs/sft.cern.ch/lcg/releases/uv/0.11.21-24fd6/x86_64-el9-gcc15-opt/bin/uv
python=/home/gwatts/.local/share/uv/python/cpython-3.14.10-linux-x86_64-gnu/bin/python3.14
base=(results/build/"$core"/wheels/*.whl)
test "${#base[@]}" -eq 1
test -s "${base[0]}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
taskset -c 5 "$build/venv/bin/python" packaging/build_demo.py "${base[0]}" "$provider" "$payload" \
  "$bundle" "$gcc" "$build/licenses" "$build/venv/bin/patchelf" --variant "atlasjets$serial" \
  --scope-index "$partition/payload-index.json" --compact-default "$compact" --reuse-nominal-nnjvt-default "$@"
taskset -c 5 .reference-venv/bin/python packaging/check_demo.py prepare "$bundle" "$installation" "$python" "$uv"
