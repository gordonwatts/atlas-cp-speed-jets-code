#!/usr/bin/env bash
set -eo pipefail

variant=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$variant/../../.." && pwd)
output=$(realpath -m "${1:?Provide a new isolated result directory}")
revision=$(cat "$variant/base-revision.txt")
mkdir "$output"
exec > "$output/build.log" 2>&1
trap 'printf "%s\n" "$?" > "$output/build-exit-code"' EXIT
source "$root/tools/reference/setup.sh"
set -u
externals="/cvmfs/atlas.cern.ch/repo/sw/software/25.2/AnalysisBaseExternals/25.2.117/InstallArea/$CMTCONFIG"
g++ --version
for arm in baseline candidate; do
  stage="$output/$arm"
  mkdir -p "$stage"/{portable,provider}
  for source in portable/random.h portable/test_resolution_cache.cxx provider/jet_smearing.h; do
    git -C "$root" show "$revision:$source" > "$stage/$source"
  done
  (cd "$stage" && sha256sum --check "$variant/base.sha256")
done
patch --directory "$output/candidate" --strip 1 --fuzz 0 < "$variant/integration.diff"
flags=(-std=c++20 -O2 -g -ffp-contract=off -frounding-math -DJETS_PORTABLE -DJETS_RNG_CACHE
       -DJETS_RNG_CACHE_DIAGNOSTICS -I"$externals/include" -L"$externals/lib" -Wl,-rpath,"$externals/lib")
printf '%s\n' "${flags[@]}" > "$output/flags.txt"
for arm in baseline candidate; do
  option=()
  if [[ $arm == candidate ]]; then option=(-DJETS_RNG_PREFIX); fi
  g++ "${flags[@]}" "${option[@]}" "$output/$arm/portable/test_resolution_cache.cxx" \
     -lMathCore -lCore -o "$output/resolution-$arm"
done
g++ "${flags[@]}" -DJETS_RNG_PREFIX -I"$output/candidate" "$variant/fixture.cxx" \
   -lMathCore -lCore -o "$output/raw-and-gaussian"
printf '%s\n' "$externals/lib:$LD_LIBRARY_PATH" > "$output/library-path.txt"
for artifact in "$output"/{resolution-baseline,resolution-candidate,raw-and-gaussian}; do
  LD_LIBRARY_PATH="$(cat "$output/library-path.txt")" ldd "$artifact" > "$artifact.ldd.txt"
done
sha256sum "$variant"/{base-revision.txt,base.sha256,integration.diff,fixture.cxx,build-fixture.sh} \
  "$output"/{baseline,candidate}/{portable/random.h,portable/test_resolution_cache.cxx,provider/jet_smearing.h} \
  "$output"/{resolution-baseline,resolution-candidate,raw-and-gaussian} > "$output/inputs.sha256"
