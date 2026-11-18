#!/usr/bin/env bash
# Isolate the generic plugin default-dispatch experiment from both pinned cores.
set -eo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
label=$1
[[ "$label" =~ ^core-plugin-batch-[a-z0-9-]+$ ]]
read -r base_commit < provenance/core-patches/derived-commit.txt
read -r base_tree < provenance/core-patches/derived-tree.txt
trial="build/$label-source"
test ! -e "$trial"
test ! -e "results/build/$label"
git clone --no-hardlinks build/core-precision "$trial"
git -C "$trial" checkout --detach "$base_commit"
test "$(git -C "$trial" write-tree)" = "$base_tree"
git -C "$trial" apply --index "$root/provenance/core-patches/trial-cache-plugin-batch-scalars.diff"
git -C "$trial" -c user.name='Jet campaign' -c user.email='jet-campaign@localhost' \
  commit -m 'Cache immutable owned columns in default plugin batch dispatch'
for module in cpp-peglib eigen lwtnn pcg-cpp pybind11 rapidjson xxhash; do
  git -C "$trial" config "submodule.$module.url" "$root/source/correctionlib/$module"
done
git -c protocol.file.allow=always -C "$trial" submodule update --init --jobs 1
bash tools/build-core.sh "$label" "$trial" "$(git -C "$trial" rev-parse HEAD)"
sha256sum "$root/provenance/core-patches/trial-cache-plugin-batch-scalars.diff" \
  > "results/build/$label/plugin-batch.patch.sha256"
