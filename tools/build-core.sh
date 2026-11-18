#!/usr/bin/env bash
# Run this script on a single build CPU, outside benchmark windows.
set -eo pipefail
checkout=$(cd "$(dirname "$0")/.." && pwd)
cd "$checkout"
run_dir=${1:?supply a new result directory below results/build}
mkdir -p results/build
mkdir "results/build/$run_dir"
log="$checkout/results/build/$run_dir"
exec > "$log/build.log" 2>&1
trap 'echo $? > "$log/exit-code"' EXIT
core_src=${2:-source/correctionlib}
core_sha=${3:-3dcfa20ec01fb40c4d5fec58383785519dec1c00}
test "$(git -C "$core_src" rev-parse HEAD)" = "$core_sha"
uv=/cvmfs/sft.cern.ch/lcg/releases/uv/0.11.21-24fd6/x86_64-el9-gcc15-opt/bin/uv
python=/home/gwatts/.local/share/uv/python/cpython-3.14.10-linux-x86_64-gnu/bin/python3.14
source /cvmfs/sft.cern.ch/lcg/releases/gcc/15.2.0-35657/x86_64-el9/setup.sh
export CMAKE_BUILD_PARALLEL_LEVEL=1 MAKEFLAGS=-j1 CMAKE_GENERATOR=Ninja
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
export UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=1 UV_CONCURRENT_DOWNLOADS=1
export CC=gcc CXX=g++
{
  uname -a
  gcc --version
  g++ --version
  "$python" -VV
  "$uv" --version
  grep Cpus_allowed_list /proc/self/status
  git -C source/correctionlib submodule status --recursive
  git -C "$core_src" rev-parse HEAD
} > "$log/toolchain.txt"
"$uv" venv --python "$python" "$log/venv"
"$uv" pip install --python "$log/venv/bin/python" --native-tls typer uproot awkward hist numpy rich 'pydantic>=2' packaging
"$uv" build --wheel --python "$log/venv/bin/python" --native-tls --out-dir "$log/wheels" "$core_src"
"$uv" pip install --python "$log/venv/bin/python" --no-deps "$log"/wheels/*.whl
"$uv" pip freeze --python "$log/venv/bin/python" > "$log/packages.txt"
sha256sum "$log"/wheels/*.whl > "$log/wheel.sha256"
"$log/venv/bin/python" -c 'import correctionlib,correctionlib._core; print(correctionlib.__version__,correctionlib._core.__file__)'
