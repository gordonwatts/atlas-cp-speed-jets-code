#!/usr/bin/env bash
set -eo pipefail
variant=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$variant/../../.." && pwd)
output=$(realpath -m "$1")
source /cvmfs/sft.cern.ch/lcg/releases/gcc/15.2.0-35657/x86_64-el9/setup.sh
mkdir "$output"
taskset -c 5 g++ -std=c++20 -O2 -ffp-contract=off -frounding-math -Wall -Wextra -Werror \
  -I"$root/source/correctionlib/rapidjson/include" -I"$root/portable" \
  "$root/portable/histogram.cxx" "$variant/fixture.cxx" -o "$output/nominal-resolution-cache-fixture"
sha256sum "$root/portable/nominal_resolution_cache.h" "$variant/fixture.cxx" "$variant/integration.diff" \
  "$root/portable/histogram.cxx" "$output/nominal-resolution-cache-fixture" > "$output/inputs.sha256"
ldd "$output/nominal-resolution-cache-fixture" > "$output/ldd.txt"
