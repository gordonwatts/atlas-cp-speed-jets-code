#!/usr/bin/env bash
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
output=$(realpath -m "$1")
source "$root/tools/reference/setup.sh"
release="$AtlasArea/InstallArea/$CMTCONFIG"
rapidjson="$root/source/correctionlib/rapidjson/include"
mkdir "$output"
taskset -c 5 g++ -std=c++23 -O2 -ffp-contract=off -Wall -Wextra -Werror \
  -I"$rapidjson" -c "$root/portable/histogram.cxx" -o "$output/histogram.o"
taskset -c 5 g++ -O2 -ffp-contract=off -Wall -Wextra -Werror \
  -DXAOD_STANDALONE -DXAOD_ANALYSIS -DROOTCORE $(root-config --cflags) \
  -I"$release/include" -I"$rapidjson" "$root/portable/lookup_replay.cxx" "$output/histogram.o" \
  -L"$release/lib" -lJetUncertaintiesLib -lAsgMessagingLib $(root-config --libs) \
  -o "$output/lookup_replay"
ldd "$output/lookup_replay" > "$output/ldd.txt"
sha256sum "$output/lookup_replay" "$release/lib/libJetUncertaintiesLib.so" \
  "$root/portable/lookup_replay.cxx" "$root/portable/histogram.cxx" \
  "$root/source/correctionlib/src/plugin.cc" > "$output/source-binaries.sha256"
