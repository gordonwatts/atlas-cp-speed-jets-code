#!/usr/bin/env bash
set -eo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
tag=${1:?fresh suffix}
[[ "$tag" =~ ^[a-z0-9-]+$ ]]
trial=provenance/provider-patches/nominal-resolution-cache
out="results/build/table-resolution-$tag"
mkdir "$out"
exec > "$out/controller.log" 2>&1
trap 'echo $? > "$out/exit-code"' EXIT
test -s portable/nominal_resolution_cache.h
rg -q 'JETS_NOMINAL_RESOLUTION_CACHE' portable/uncertainty.cxx
taskset -c 5 bash tools/build-portable.sh "joint-resolution-$tag" core-precision-002 portable-lwtnn-001 rng-cache nominal-resolution-cache
taskset -c 5 bash "$trial/build-fixture.sh" "$out/fixture"
gcc=/cvmfs/sft.cern.ch/lcg/releases/gcc/15.2.0-35657/x86_64-el9/setup.sh
taskset -c 3 env -i PATH=/usr/bin:/bin LC_ALL=C bash -c 'source "$1" && exec "$2"' fixture "$gcc" "$root/$out/fixture/nominal-resolution-cache-fixture" > "$out/fixture.log"
