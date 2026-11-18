#!/usr/bin/env bash
set -eo pipefail

variant=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$variant/../../.." && pwd)
output=$(realpath -m "${1:?Provide a new isolated result directory}")
payload=$(realpath -e "${2:?Provide the approved canonical jets.json}")
sdk=$(realpath -e "${3:-"$root/results/build/core-precision-002/venv/lib/python3.14/site-packages/correctionlib"}")
lwtnn_build=$(realpath -e "${4:-"$root/results/build/portable-lwtnn-001"}")
lwtnn="$lwtnn_build/build/lib/liblwtnn-stat.a"
python=$(realpath -s "$sdk/../../../../bin/python")

mkdir "$output"
exec > "$output/build.log" 2>&1
trap 'printf "%s\n" "$?" > "$output/build-exit-code"' EXIT
source "$root/tools/reference/setup.sh"
set -u
externals="/cvmfs/atlas.cern.ch/repo/sw/software/25.2/AnalysisBaseExternals/25.2.117/InstallArea/$CMTCONFIG"
core="$sdk/lib/libcorrectionlib.so"
rapid="$root/source/correctionlib/rapidjson/include"
g++ --version
test -s "$lwtnn"
test -x "$python"

revision=$(cat "$variant/base-revision.txt")
for arm in baseline candidate; do
  stage="$output/$arm"
  mkdir -p "$stage/provider" "$stage/portable"
  git -C "$root" show "$revision:provider/small_jvt.cxx" > "$stage/provider/small_jvt.cxx"
  git -C "$root" show "$revision:provider/small_jvt.h" > "$stage/provider/small_jvt.h"
  git -C "$root" show "$revision:portable/lwtnn.h" > "$stage/portable/lwtnn.h"
  (cd "$stage" && sha256sum --check "$variant/base.sha256")
  cp "$variant/plugin.cxx" "$stage/provider/plugin.cxx"
done
patch --directory "$output/candidate" --strip 1 --fuzz 0 < "$variant/integration.diff"

flags=(-DJETS_PORTABLE -std=c++20 -O2 -g -fPIC -ffp-contract=off -fvisibility=hidden
       -I"$sdk/include" -I"$rapid" -I"$externals/include" -I"$externals/include/eigen3")
printf '%s\n' "${flags[@]}" > "$output/provider-flags.txt"
for arm in baseline candidate; do
  choice=()
  if [[ $arm == candidate ]]; then choice=(-DJETS_NNJVT_REUSE_INPUTS); fi
  g++ "${flags[@]}" "${choice[@]}" -shared -Wl,-z,defs -Wl,-rpath,"$sdk/lib" \
    "$output/$arm/provider/plugin.cxx" -Wl,--exclude-libs,liblwtnn-stat.a "$lwtnn" \
    -L"$sdk/lib" -lcorrectionlib -o "$output/$arm.so"
done
g++ -std=c++20 -O2 -g -ffp-contract=off -frounding-math -Wall -Wextra -Werror -pthread \
  -I"$sdk/include" -I"$rapid" "$variant/fixture.cxx" -L"$sdk/lib" -lcorrectionlib -ldl \
  -Wl,-rpath,"$sdk/lib" -o "$output/nnjvt-inputs-fixture"
PYTHONDONTWRITEBYTECODE=1 "$python" "$variant/prepare-payload.py" "$payload" "$output"
export LD_LIBRARY_PATH="$sdk/lib:$externals/lib:$LD_LIBRARY_PATH"
printf '%s\n' "$LD_LIBRARY_PATH" > "$output/library-path.txt"
for binary in "$output"/{baseline.so,candidate.so,nnjvt-inputs-fixture}; do
  ldd "$binary" > "$binary.ldd.txt"
  if grep -E 'not found|lib(Core|RIO|Hist|Tree|Physics|MathCore|xAOD|JetCalib|JetUncertainties|ROOT|Asg|lwtnn.so)' "$binary.ldd.txt"; then
    exit 1
  fi
done
sha256sum "$variant"/{base-revision.txt,base.sha256,integration.diff,fixture.cxx,plugin.cxx,prepare-payload.py,build-fixture.sh} \
  "$output"/{baseline,candidate}/provider/{small_jvt.cxx,small_jvt.h,plugin.cxx} \
  "$output"/{baseline,candidate}/portable/lwtnn.h \
  "$externals/include/lwtnn/generic/FastGraph."{hh,tcc} "$core" "$lwtnn" "$payload" \
  "$output"/{baseline.so,candidate.so,nnjvt-inputs-fixture,nnjvt-update.json,cuts-and-network.json,model-receipt.json} \
  > "$output/inputs.sha256"
