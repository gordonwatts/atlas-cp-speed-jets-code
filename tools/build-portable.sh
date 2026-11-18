#!/usr/bin/env bash
set -eo pipefail

if [[ -z ${JETS_PORTABLE_BUILD_SNAPSHOT:-} ]]; then
  export JETS_PORTABLE_BUILD_SNAPSHOT=1
  exec bash -c "$(cat -- "$0")" "$0" "$@"
fi

root=$(cd "$(dirname "$0")/.." && pwd)
label=${1:?pass a new build name}
core=${2:-core-precision-002}
lwtnn_build=${3:-portable-lwtnn-001}
jets_candidate_choices=()
jets_candidate_defines=()
for choice in "${@:4}"; do
  case " ${jets_candidate_choices[*]} " in
    *" $choice "*) echo "Duplicate provider build choice: $choice" >&2; exit 2 ;;
  esac
  case "$choice" in
    rng-cache) jets_candidate_defines+=(-DJETS_RNG_CACHE) ;;
    rng-cache-large) jets_candidate_defines+=(-DJETS_RNG_CACHE -DJETS_RNG_CACHE_LARGE) ;;
    rng-cache-diagnostics) jets_candidate_defines+=(-DJETS_RNG_CACHE_DIAGNOSTICS) ;;
    large-jmr-match-cache) jets_candidate_defines+=(-DJETS_LARGE_JMR_MATCH_CACHE) ;;
    uncertainty-batch) jets_candidate_defines+=(-DJETS_UNCERTAINTY_BATCH) ;;
    fast-variable-flow) jets_candidate_defines+=(-DJETS_FAST_VARIABLE_FLOW) ;;
    *) echo "Unknown provider build choice: $choice" >&2; exit 2 ;;
  esac
  jets_candidate_choices+=("$choice")
done
rng_cache_slots=0
case " ${jets_candidate_choices[*]} " in
  *" rng-cache "*) rng_cache_slots=65536 ;;
esac
case " ${jets_candidate_choices[*]} " in
  *" rng-cache-large "*)
    if (( rng_cache_slots != 0 )); then
      echo "Choose exactly one Gaussian cache size" >&2
      exit 2
    fi
    rng_cache_slots=262144 ;;
esac
mkdir "$root/results/build/$label"
cd "$root/results/build/$label"
exec > build.log 2>&1
trap 'echo $? > exit-code' EXIT
source "$root/tools/reference/setup.sh"
set -u
externals="/cvmfs/atlas.cern.ch/repo/sw/software/25.2/AnalysisBaseExternals/25.2.117/InstallArea/$CMTCONFIG"
sdk="$root/results/build/$core/venv/lib/python3.14/site-packages/correctionlib"
lwtnn="$root/results/build/$lwtnn_build/build/lib/liblwtnn-stat.a"
test -s "$lwtnn"
cxx_flags=(-DJETS_PORTABLE "${jets_candidate_defines[@]}" -std=c++20 -O2 -g -fPIC -ffp-contract=off
           -I"$sdk/include" -I"$root/source/correctionlib/rapidjson/include" -I"$root/provider"
           -I"$externals/include" -I"$externals/include/eigen3")
sources=(provider/registry provider/eta_jes provider/large_jms provider/large_jmr provider/small_jvt
         provider/small_pileup provider/small_gsc provider/event_inputs provider/uncertainty_node
         portable/histogram portable/uncertainty)
if [[ " ${jets_candidate_choices[*]} " == *" rng-cache-diagnostics "* ]]; then
  sources+=(provider/rng_cache_diagnostics)
fi
printf '%s\n' "${jets_candidate_choices[@]:-baseline}" > build-choices.txt
printf '%d\n' "$rng_cache_slots" > rng-cache-slots.txt
printf '%s\n' "-DJETS_PORTABLE" "${jets_candidate_defines[@]}" > build-defines.txt
printf '%s\n' "${cxx_flags[@]}" > compile-flags.txt
git -C "$root" rev-parse HEAD > source-revision.txt
printf '%s\n' "$BASH_EXECUTION_STRING" > launched-script.sh
sha256sum launched-script.sh > launched-script.sha256
source_files=()
for src in "${sources[@]}"; do source_files+=("$root/$src.cxx"); done
sha256sum "$root/tools/build-portable.sh" "${source_files[@]}" \
  "$root"/provider/*.h "$root"/portable/*.h > source-inputs.sha256
g++ --version
grep Cpus_allowed_list /proc/self/status
for src in "${sources[@]}"; do
  echo "Compiling $src"
  local_flags=()
  if [[ "$src" == provider/small_jvt ]]; then local_flags+=(-fvisibility=hidden); fi
  g++ "${cxx_flags[@]}" "${local_flags[@]}" -MD -MF "$(basename "$src").d" -c "$root/$src.cxx" -o "$(basename "$src").o"
done
g++ -shared -Wl,-z,defs -Wl,-rpath,'$ORIGIN' -o libjets.so *.o \
  -Wl,--exclude-libs,liblwtnn-stat.a "$lwtnn" -L"$sdk/lib" -lcorrectionlib
readelf -d libjets.so > needed.txt
LD_LIBRARY_PATH="$sdk/lib:$externals/lib:$LD_LIBRARY_PATH" ldd libjets.so > libraries.txt
if grep -E 'lib(Core|RIO|Hist|Tree|Physics|MathCore|xAOD|JetCalib|JetUncertainties|ROOT|Asg)' libraries.txt; then
  echo "Unexpected ROOT/AnalysisBase runtime in portable build" >&2
  exit 1
fi
sha256sum libjets.so "$sdk/lib/libcorrectionlib.so" "$lwtnn" > native.sha256
