#!/usr/bin/env bash
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
output=$(realpath -m "$1")
source "$root/tools/reference/setup.sh"
rel="$AtlasArea/InstallArea/$CMTCONFIG"
rapidjson="$root/source/correctionlib/rapidjson/include"
mkdir -p "$output"
taskset -c 5 g++ -std=c++23 -O2 -ffp-contract=off -Wall -Wextra -Werror \
  -I"$rapidjson" -c "$root/portable/histogram.cxx" -o "$output/histogram.o"
taskset -c 5 g++ -O2 -ffp-contract=off -DXAOD_STANDALONE -DXAOD_ANALYSIS -DROOTCORE \
  $(root-config --cflags) -I"$rel/include" -I"$rapidjson" \
  "$root/portable/test_histogram.cxx" "$output/histogram.o" \
  -L"$rel/lib" -lJetCalibToolsLib -lJetUncertaintiesLib -lAsgMessagingLib \
  $(root-config --libs) -o "$output/test_histogram"
