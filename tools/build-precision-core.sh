#!/usr/bin/env bash
# Create a separate source checkout and wheel; the comparison core stays intact.
set -eo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
label=${1:?new core build label}
dst="build/$label-source"
test ! -e "$dst"
git clone --no-hardlinks source/correctionlib "$dst"
git -C "$dst" checkout --detach 3dcfa20ec01fb40c4d5fec58383785519dec1c00
git -C "$dst" apply --index "$root/provenance/core-patches/develop-full-precision.diff"
# Reproduce the numeric source tree independently of the local commit's timestamp.
expected=$(cat provenance/core-patches/derived-tree.txt)
test "$(git -C "$dst" write-tree)" = "$expected"
git -C "$dst" -c user.name='Jet campaign' -c user.email='jet-campaign@localhost' commit -m 'Apply pinned full-precision parsing fix to the develop evaluator'
for module in cpp-peglib eigen lwtnn pcg-cpp pybind11 rapidjson xxhash; do
  git -C "$dst" config "submodule.$module.url" "$root/source/correctionlib/$module"
done
git -c protocol.file.allow=always -C "$dst" submodule update --init --jobs 1
bash tools/build-core.sh "$label" "$dst" "$(git -C "$dst" rev-parse HEAD)"
