#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
label=${1:?supply a new build result name}
if [[ ! "$label" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
  echo "The build result name must be a single directory name" >&2
  exit 2
fi
sdk=${2:-"$root/results/build/core-001/venv/lib/python3.14/site-packages/correctionlib"}
sdk=$(realpath "$sdk")
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
python "$root/framework/prepare.py" "$release" "$dest" "$root/framework"
cmake -S src/Projects/WorkDir -B build -DATLAS_PROJECT=AnalysisBase \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -Dcorrectionlib_DIR="$sdk/cmake"
cmake --build build --target JetCalibToolsLib --parallel 1

# Only the rebuilt implementation and its CorrectionLib client enter the
# runtime library search path. Component factories, dictionaries and Python
# modules come from the pinned ABI-compatible release.
mkdir overlay
for library in libJetCalibToolsLib.so libJetsCorrectionRuntime.so; do
  cp -L "build/$CMTCONFIG/lib/$library" "overlay/$library"
done
for library in overlay/*.so; do
  sha256sum "$library"
  readelf -d "$library"
  ldd "$library"
done > overlay-libraries.txt
