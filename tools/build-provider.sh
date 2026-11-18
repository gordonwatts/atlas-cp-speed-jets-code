#!/usr/bin/env bash
set -eo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
label=${1:?supply a new build result directory name}
core=${2:-core-001}
mkdir -p results/build
mkdir "results/build/$label"
dest="$root/results/build/$label"
cd "$dest"
exec > build.log 2>&1
trap 'echo $? > exit-code' EXIT
source "$root/tools/reference/setup.sh"
set -u
release="$AtlasArea/InstallArea/$CMTCONFIG"
externals="/cvmfs/atlas.cern.ch/repo/sw/software/25.2/AnalysisBaseExternals/25.2.117/InstallArea/$CMTCONFIG"
sdk="$root/results/build/$core/venv/lib/python3.14/site-packages/correctionlib"
sha256sum "$sdk/lib/libcorrectionlib.so" "$root"/provider/*.cxx "$root"/provider/*.h > build-inputs.sha256
echo "Build affinity: $(grep Cpus_allowed_list /proc/self/status)"
cxx_flags=(-DXAOD_STANDALONE -DXAOD_ANALYSIS -DROOTCORE -O2 -g -fPIC -ffp-contract=off -I"$sdk/include" -I"$root/source/correctionlib/rapidjson/include"
           -I"$root/provider" -I"$release/include" -I"$externals/include" -I"$externals/include/eigen3")
sources=(registry histogram eta_jes large_jms large_jmr small_jvt small_pileup small_gsc event_inputs uncertainty_components jet_uncertainty)
for item in "${sources[@]}"; do
  echo "Compiling $item"
  g++ $(root-config --cflags) "${cxx_flags[@]}" -c "$root/provider/$item.cxx" -o "$item.o"
done
g++ -shared -Wl,-z,defs -o libjets.so *.o -L"$sdk/lib" -Wl,-rpath,"$sdk/lib" -lcorrectionlib \
  -L"$release/lib" -lJetCalibToolsLib -lJetUncertaintiesLib -lPATInterfaces -lxAODJet -lxAODEventInfo -lAthContainers -lAsgMessagingLib -L"$externals/lib" -llwtnn $(root-config --libs)
ldd libjets.so > link-libraries.txt
sha256sum libjets.so > provider.sha256
