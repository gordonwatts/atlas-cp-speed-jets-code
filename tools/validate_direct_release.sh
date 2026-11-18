#!/usr/bin/env bash
# Re-evaluate the original PHYSLITE, then reconcile the original public energy/alias fields.
set -eo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
py=${1:?analysis Python installation}
provider=$(realpath "${2:?provider}")
payload=$(realpath "${3:?payload}")
out=$(realpath -m "${4:?fresh output directory}")
scope=${5:?small, large, or both}
shift 5
case "$scope" in small|large|both) ;; *) exit 2;; esac
for choice in "$@"; do
  case "$choice" in --reuse-nominal-nnjvt-update) ;; *) exit 2;; esac
done
[[ $(awk '/^Cpus_allowed_list:/ {print $2}' /proc/$$/status) == 3 ]]
mkdir "$out"
exec > "$out/controller.log" 2>&1
trap 'echo $? > "$out/exit-code"' EXIT
for setting in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS BLIS_NUM_THREADS ROOT_MAX_THREADS; do
  export "$setting=1"
done
unset PYTHONHOME PYTHONPATH
site=$("$py" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
packaged_lib="$site/correctionlib/lib"
if [[ "$provider" == "$packaged_lib/libjets.so" ]]; then
  # The checkout observer imports Awkward before CorrectionLib. Match the
  # visible script bootstrap before either library can load the system C++ ABI.
  test -s "$packaged_lib/libstdc++.so.6"
  export LD_LIBRARY_PATH="$packaged_lib"
fi
printf 'LD_LIBRARY_PATH=%s\n' "${LD_LIBRARY_PATH:-}" > "$out/diagnostic-environment.txt"
original=$(sed -n 's/^export JETS_REFERENCE_INPUT=//p' tools/reference/setup.sh)
stock="$root/results/reference/full-clean-001/reference.root"
energy="$root/results/build/columnar-alias-energy-portable-001/libcolumnar-alias-energy.so"
provenance="$(dirname "$payload")/provenance.json"
extra=()
if [[ $scope == both ]]; then
  extra+=(--distributions "$out/distributions" --energy-library "$energy")
fi
command=("$py" -u tools/compare_columnar.py "$original" "$stock" "$out/comparison.json"
         --scope "$scope" --mode full --events 20000 --chunk 128 --provider "$provider"
         --payload "$payload" --provenance "$provenance" "${extra[@]}" "$@")
printf '%q ' "${command[@]}" > "$out/command.txt"
printf '\n' >> "$out/command.txt"
sha256sum "$provider" "$payload" "$energy" columnar/{jets,inputs,distributions}.py tools/compare_columnar.py > "$out/inputs.sha256"
git rev-parse HEAD > "$out/revision"
(time "${command[@]}" > "$out/compare.log" 2>&1) 2> "$out/comparison.time"
if [[ $scope == both ]]; then
  (time "$py" tools/compare_columnar_aliases.py "$out/comparison.json" "$out/observer-aliases.json" \
    --energy-library "$energy" > "$out/aliases.log" 2>&1) 2> "$out/aliases.time"
fi
