#!/usr/bin/env bash
# Compile after applying integration.diff in the isolated trial checkout.
set -eo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$root"
dest=$(realpath -m "${1:?fresh fixture output directory}")
payload=$(realpath "${2:?archived complete packed CorrectionSet}")
mkdir "$dest"
exec > "$dest/build.log" 2>&1
trap 'echo $? > "$dest/exit-code"' EXIT
source tools/reference/setup.sh
rel="$AtlasArea/InstallArea/$CMTCONFIG"
rapid="$root/source/correctionlib/rapidjson/include"
common=(-std=c++23 -O2 -ffp-contract=off -I"$rapid" -I"$root/portable")
atlas=(-DXAOD_STANDALONE -DXAOD_ANALYSIS -DROOTCORE -I"$rel/include")
read -ra root_flags <<< "$(root-config --cflags)"
read -ra root_libs <<< "$(root-config --libs)"
dir=provenance/provider-patches/variable-axis-index
python3 "$dir/geometry.py" "$payload" > "$dest/geometry.txt"
for mode in off on; do
  choice=(-DJETS_FAST_VARIABLE_FLOW)
  if [[ $mode == on ]]; then choice+=(-DJETS_VARIABLE_AXIS_INDEX); fi
  g++ "${common[@]}" "${choice[@]}" -c portable/histogram.cxx -o "$dest/hist-$mode.o"
  g++ "${common[@]}" "${choice[@]}" "$dir/fixture.cxx" "$dest/hist-$mode.o" -o "$dest/axis-$mode"
  g++ "${common[@]}" "${choice[@]}" "${atlas[@]}" "${root_flags[@]}" portable/test_histogram.cxx \
    "$dest/hist-$mode.o" -L"$rel/lib" -lJetCalibToolsLib -lJetUncertaintiesLib -lAsgMessagingLib \
    "${root_libs[@]}" -o "$dest/oracle-$mode"
done
sha256sum "$payload" "$dest"/{geometry.txt,axis-off,axis-on,oracle-off,oracle-on} > "$dest/inputs.sha256"
