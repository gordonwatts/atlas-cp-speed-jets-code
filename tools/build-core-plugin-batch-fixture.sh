#!/usr/bin/env bash
# Compile the same fixture against each SDK. Run the resulting binaries separately on CPU3.
set -eo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
sdk=$(realpath "$1")
output=$(realpath -m "$2")
source /cvmfs/sft.cern.ch/lcg/releases/gcc/15.2.0-35657/x86_64-el9/setup.sh
mkdir "$output"
taskset -c 5 g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  "$root/tools/core-plugin-batch-fixture.cxx" -I"$sdk/include" -L"$sdk/lib" \
  -Wl,-rpath,"$sdk/lib" -lcorrectionlib -o "$output/core-plugin-batch-fixture"
ldd "$output/core-plugin-batch-fixture" > "$output/ldd.txt"
sha256sum "$output/core-plugin-batch-fixture" "$sdk/lib/libcorrectionlib.so" \
  "$root/tools/core-plugin-batch-fixture.cxx" > "$output/fixture.sha256"
