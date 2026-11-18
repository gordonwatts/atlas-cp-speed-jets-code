#!/usr/bin/env bash
# Run a bounded independent check before the full-file one-change comparisons.
set -eo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
plan=$(realpath "${1:?supply the one-change plan}")
result=$(realpath -m "${2:?supply an unused screening directory}")
mode=${3:-all}
[[ $mode == all || $mode == objects ]]
if [[ $(awk '/^Cpus_allowed_list:/ {print $2}' /proc/$$/status) != 3 ]]; then
  echo "Reserve CPU 3 for this sequential screen" >&2
  exit 2
fi
py="$root/.reference-venv/bin/python"
mkdir "$result"
exec > "$result/controller.log" 2>&1
trap 'echo $? > "$result/exit-code"' EXIT
for key in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS BLIS_NUM_THREADS ROOT_MAX_THREADS RAYON_NUM_THREADS; do
  export "$key=1"
done
unset PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1
if "$py" -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1])).get("compiler_only_runtime"))' "$plan"; then
  gcc_lib=/cvmfs/sft.cern.ch/lcg/releases/gcc/15.2.0-35657/x86_64-el9/lib64
  test -s "$gcc_lib/libstdc++.so.6"
  test -s "$gcc_lib/libgcc_s.so.1"
  unset LD_PRELOAD
  export LD_LIBRARY_PATH="$gcc_lib"
else
  source tools/reference/setup.sh
  unset PYTHONPATH PYTHONHOME
fi
"$py" - "$root" "$plan" "$result" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

from tools.time_columnar_trials import hashed, runtime_environment, variant_runtime, verify_script_sources

root, plan, output = map(Path, sys.argv[1:])
config = json.loads(plan.read_text())
verify_script_sources(config)
authority = config.get("observer", str(root / "results/reference/full-clean-001/reference.root"))
all_scopes = config.get("screen_each_scope", False)
events = config.get("screen_events", 32)
observed_branches = {}
receipts = []

for name, entry in config["variants"].items():
    if name == "portable" and not all_scopes:
        continue
    scopes = entry.get("scopes", config["scopes"])
    for scope in (scopes if all_scopes or "both" not in scopes else ["both"]):
        result = output / f"objects-{name}-{scope}.json"
        runtime = variant_runtime(config, entry, scope)
        command = [runtime["python"], str(root / "tools/compare_columnar.py"), config["input"], authority,
                   str(result), "--scope", scope, "--mode", "full", "--provider", entry["provider"],
                   "--payload", runtime["payload"], "--provenance", runtime["provenance"],
                   "--events", str(events), "--chunk", "8", *entry.get("options", [])]
        verify_script_sources(config)
        print(command, flush=True)
        with (output / f"objects-{name}-{scope}.log").open("x") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                           env=runtime_environment(dict(os.environ), runtime["sdk"]), check=True)
        proof = json.loads(result.read_text())
        if not proof["passed"] or proof["entries_compared"] != events:
            raise RuntimeError(f"Independent comparison failed for {name}/{scope}")
        if "input_identity" in config and proof["source_file_uuid"] != config["input_identity"]["tfile_uuid"]:
            raise RuntimeError(f"Original PHYSLITE identity changed for {name}/{scope}")
        if runtime["sdk"]:
            expected_core = (Path(runtime["sdk"]) / "lib/libcorrectionlib.so").resolve()
            if Path(proof["runtime"]["core"]).resolve() != expected_core:
                raise RuntimeError(f"Unexpected mapped core for {name}/{scope}")
        expected_provider = Path(entry["provider"]).resolve()
        if Path(proof["runtime"]["provider"]).resolve() != expected_provider:
            raise RuntimeError(f"Unexpected mapped provider for {name}/{scope}")
        if config.get("require_matching_branches"):
            branches = proof["source_branches_read"]
            if observed_branches.setdefault(scope, branches) != branches:
                raise RuntimeError(f"Original branch request changed for {name}/{scope}")
            excluded = {
                "small": ("AnalysisLargeRJetsAux", "AntiKt10TruthSoftDropBeta100Zcut10JetsAux"),
                "large": ("AnalysisJetsAux", "Kt4EMPFlowNeutEventShapeAux"),
                "both": (),
            }
            if any(branch.startswith(excluded[scope]) for branch in branches):
                raise RuntimeError(f"Scoped job read a branch outside {scope}: {name}")
        receipts.append({"variant": name, "scope": scope, "proof": str(result), **runtime,
                         "payload_sha256": hashed(runtime["payload"]),
                         "provenance_sha256": hashed(runtime["provenance"])})

with (output / "object-screen.json").open("x") as stream:
    json.dump({"passed": True, "entries_per_case": events, "cases": receipts,
               "source_branches_by_scope": observed_branches}, stream, indent=2)
    stream.write("\n")
PY
if [[ $mode == all ]]; then
  "$py" tools/time_columnar_trials.py "$plan" "$result/workflow"
fi
