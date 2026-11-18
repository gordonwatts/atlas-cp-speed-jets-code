#!/usr/bin/env bash
# Compile the independent and differential checks. Execution uses the reserved analysis CPU later.
set -eo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
dest=$(realpath -m "${1:?new fixture directory}")
jmr=$(realpath "${2:?candidate provider enabling large-jmr-match-cache}")
sdk=$(realpath "${3:?matching CorrectionLib SDK}")
mkdir "$dest"
exec > "$dest/build.log" 2>&1
trap 'echo $? > "$dest/exit-code"' EXIT
source tools/reference/setup.sh
rel="$AtlasArea/InstallArea/$CMTCONFIG"
rapid="$root/source/correctionlib/rapidjson/include"
common=(-std=c++23 -O2 -ffp-contract=off -I"$rapid" -I"$root/provider")
atlas=(-DXAOD_STANDALONE -DXAOD_ANALYSIS -DROOTCORE -I"$rel/include")
read -ra root_flags <<< "$(root-config --cflags)"
read -ra root_libs <<< "$(root-config --libs)"
for mode in plain flow; do
  choice=()
  if [[ $mode == flow ]]; then choice=(-DJETS_FAST_VARIABLE_FLOW); fi
  g++ "${common[@]}" "${choice[@]}" -c portable/histogram.cxx -o "$dest/hist-$mode.o"
  g++ "${common[@]}" "${choice[@]}" portable/test_variable_flow.cxx "$dest/hist-$mode.o" -o "$dest/flow-$mode"
  g++ "${common[@]}" "${atlas[@]}" "${root_flags[@]}" portable/test_histogram.cxx "$dest/hist-$mode.o" \
    -L"$rel/lib" -lJetCalibToolsLib -lJetUncertaintiesLib -lAsgMessagingLib "${root_libs[@]}" -o "$dest/oracle-$mode"
done
g++ "${common[@]}" -DJETS_PORTABLE provider/test_packed_histogram.cxx "$dest/hist-plain.o" -o "$dest/packed-portable"
g++ "${common[@]}" "${atlas[@]}" "${root_flags[@]}" provider/test_packed_histogram.cxx provider/histogram.cxx \
  -L"$rel/lib" -lJetCalibToolsLib -lJetUncertaintiesLib "${root_libs[@]}" -o "$dest/packed-stock"
g++ "${common[@]}" "${root_flags[@]}" -DJETS_PORTABLE -DJETS_RNG_CACHE -DJETS_RNG_CACHE_LARGE \
  -DJETS_RNG_CACHE_DIAGNOSTICS portable/test_resolution_cache.cxx "${root_libs[@]}" -o "$dest/rng-large"
g++ "${common[@]}" -DJETS_PORTABLE -DJETS_LARGE_JMR_MATCH_CACHE -I"$sdk/include" \
   portable/test_jmr_match_cache.cxx "$jmr" -L"$sdk/lib" -lcorrectionlib \
   -Wl,-rpath,"$sdk/lib" -Wl,-rpath,"$(dirname "$jmr")" -o "$dest/jmr-match"
sha256sum "$dest"/{flow-plain,flow-flow,oracle-plain,oracle-flow,packed-portable,packed-stock,rng-large,jmr-match} > "$dest/binaries.sha256"
