#!/usr/bin/env bash
set -euo pipefail

main() {
root=$(cd "$(dirname "$0")/.." && pwd)
label=${1:?supply a new build result name}
if [[ ! "$label" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "The build result name must be a single directory name" >&2
  exit 2
fi
sdk=${2:-"$root/results/build/core-001/venv/lib/python3.14/site-packages/correctionlib"}
sdk=$(realpath "$sdk")
jets_framework_profile=${3:-nominal}
jets_framework_choice=${4:-baseline}
jets_framework_cmake_options=()
case "$jets_framework_choice" in
  baseline) ;;
  reuse-inputs)
    if [[ "$jets_framework_profile" != full ]]; then
      echo "Input reuse requires the full framework profile" >&2
      exit 2
    fi
    jets_framework_cmake_options+=(-DJETS_FRAMEWORK_REUSE_INPUTS=ON)
    ;;
  *) echo "Unknown framework build choice $jets_framework_choice" >&2; exit 2 ;;
esac
jets_framework_targets=(JetCalibToolsLib)
jets_framework_libraries=(libJetCalibToolsLib.so libJetsCorrectionRuntime.so)
case "$jets_framework_profile" in
  nominal|jvt|full) ;;
  *)
    echo "Unknown framework build profile $jets_framework_profile" >&2
    exit 2
    ;;
esac
if [[ "$jets_framework_profile" != nominal ]]; then
  jets_framework_targets+=(JetJvtEfficiencyLib)
  jets_framework_libraries+=(libJetMomentToolsLib.so libJetJvtEfficiencyLib.so)
fi
if [[ "$jets_framework_profile" == full ]]; then
  jets_framework_targets+=(JetUncertaintiesLib AsgAnalysisAlgorithmsLib)
  jets_framework_libraries+=(libJetUncertaintiesLib.so libParticleJetToolsLib.so libAsgAnalysisAlgorithmsLib.so)
fi
mkdir -p "$root/results/build"
dest="$root/results/build/$label"
mkdir "$dest"
cd "$dest"
exec > build.log 2>&1
trap 'echo $? > exit-code' EXIT

set +eu
source "$root/tools/reference/setup.sh"
release_setup_status=$?
set -eu
test "$release_setup_status" -eq 0
release="$AtlasArea/InstallArea/$CMTCONFIG"
test "$CMTCONFIG" = x86_64-el9-gcc15-opt
test -s "$sdk/lib/libcorrectionlib.so"
export CMAKE_BUILD_PARALLEL_LEVEL=1 MAKEFLAGS=-j1
grep Cpus_allowed_list /proc/self/status
sha256sum "$sdk/lib/libcorrectionlib.so" > evaluator.sha256
printf '%s\n' "$jets_framework_choice" > build-choice.txt
printf '%s\n' "${jets_framework_cmake_options[@]}" > candidate-cmake-flags.txt
git -C "$root" rev-parse HEAD > source-revision.txt
find "$root/framework" -type f \( -name 'CMakeLists.txt' -o -name '*.py' -o -name '*.h' -o -name '*.icc' \
  -o -name '*.inc' -o -name '*.cxx' \) -print0 | sort -z | xargs -0 sha256sum > framework-source.sha256
python "$root/framework/prepare.py" "$release" "$dest" "$root/framework" "$jets_framework_profile"
cmake -S src/Projects/WorkDir -B build -DATLAS_PROJECT=AnalysisBase \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -Dcorrectionlib_DIR="$sdk/cmake" "${jets_framework_cmake_options[@]}"
cmake --build build --target "${jets_framework_targets[@]}" --parallel 1

# Only the rebuilt implementation and its CorrectionLib client enter the
# runtime library search path. Component factories, dictionaries and Python
# modules come from the pinned ABI-compatible release.
mkdir overlay
for library in "${jets_framework_libraries[@]}"; do
  cp -L "build/$CMTCONFIG/lib/$library" "overlay/$library"
done
for library in overlay/*.so; do
  sha256sum "$library"
  readelf -d "$library"
  LD_LIBRARY_PATH="$dest/overlay:$sdk/lib:$LD_LIBRARY_PATH" ldd "$library"
done > overlay-libraries.txt
}

main "$@"
