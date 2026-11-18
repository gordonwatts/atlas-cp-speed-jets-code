#!/usr/bin/env bash
set -e
here=$(cd "$(dirname "$0")" && pwd)
release="$AtlasArea/InstallArea/$CMTCONFIG"
target="$here/../../.reference-build"
mkdir -p "$target"
taskset -c 3 g++ -shared -fPIC -O2 -DXAOD_STANDALONE \
    $(root-config --cflags) -I"$release/include" "$here/ObserveValidity.cxx" \
    -L"$release/lib" -lAnaAlgorithmLib -lAsgTools -lSystematicsHandlesLib \
    -lxAODJet -lAthContainers $(root-config --libs) \
    -o "$target/libJetsReferenceObserver.so"
