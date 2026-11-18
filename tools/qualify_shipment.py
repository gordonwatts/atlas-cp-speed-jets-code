"""Run a named end-to-end validation step against one frozen jet shipment."""
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Annotated

import typer

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results/reference"
REF_PYTHON = ROOT / ".reference-venv/bin/python"
SCOPES = ("small", "large", "both")
THREAD_LIMITS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "BLIS_NUM_THREADS", "ROOT_MAX_THREADS", "RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def main(specification: Path, stage: str, scope: str = "both",
         jobs: Annotated[int | None, typer.Option(min=1)] = None):
    """Run on CPU 3. Each step's own tool writes its detailed immutable receipts."""
    if os.sched_getaffinity(0) != {3} or len(list(Path("/proc/self/task").iterdir())) != 1:
        raise ValueError("Launch the controller with one thread on CPU 3")
    plan = json.loads(specification.read_text())
    if scope not in SCOPES:
        raise ValueError(f"Unknown scope {scope}")
    paths = {key: (ROOT / plan[key]).resolve(strict=True) for key in
             ("overlay", "provider", "payload", "sdk", "bundle", "installation", "partition")}
    output = (ROOT / plan["output"]).resolve()
    output.mkdir(exist_ok=True)
    original = Path(json.loads((ROOT / "config/jets-input.json").read_text())["path"]).resolve(strict=True)
    bundle = paths["bundle"]
    manifest = json.loads((bundle / "bundle.json").read_text())
    defaults = manifest["demo_defaults"]
    compact = defaults["compact_payload_default_scopes"]
    if not set(compact) <= {"small", "large"}:
        raise ValueError(f"Unknown default scopes: {compact}")
    python = paths["installation"] / "pip/env/bin/python"
    site = python.parents[1] / "lib/python3.14/site-packages"
    packaged_provider = site / "correctionlib/lib/libjets.so"
    packaged_payload = site / "atlas_jet_demo/data/jets.json"
    script = bundle / manifest["visible_script"]["name"]
    wheel = bundle / manifest["wheel"]["name"]
    prefix = "combined" if scope == "both" else scope
    stock = REFERENCE / f"full-file-pilot-{prefix}-002"
    histogram = stock / "01-full/histograms.npz"
    framework = output / "B-combined"
    combined_c = output / "C-both"
    env = {k: v for k, v in os.environ.items() if k not in
           {"PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD", "JETS_PROVIDER", "JETS_PAYLOAD"}}
    env.update(dict.fromkeys(THREAD_LIMITS, "1"))

    def run(*args, check=True):
        return subprocess.run([str(arg) for arg in args], cwd=ROOT, env=env, check=check).returncode

    if stage == "objects":
        payload = packaged_payload if scope == "both" or scope not in compact else (
            paths["partition"] / "scopes" / scope / "jets.json")
        options = ["--reuse-nominal-nnjvt-update"] if defaults["reuse_nominal_nnjvt_update"] else []
        run("bash", "tools/validate_direct_release.sh", python, packaged_provider,
            payload, output / f"C-{scope}", scope, *options)
    elif stage == "shipped":
        run(REF_PYTHON, "packaging/compare_scoped_demo.py", "check", paths["installation"],
            original, histogram, scope, output / f"shipped-{scope}",
            "--events", "20000", "--verify-projection-identity")
    elif stage == "framework":
        if scope == "both":
            run("bash", "tools/verify-framework.sh", paths["overlay"], paths["provider"], paths["payload"],
                paths["sdk"], framework, REFERENCE / "full-clean-001/reference.root", histogram)
        else:
            payload = paths["partition"] / "scopes" / scope / "jets.json" if scope in compact else paths["payload"]
            pilot = output / f"B-{scope}"
            # A rejected timing audit can still leave all three complete outputs;
            # the independent recovery verifies every branch before accepting physics.
            if not pilot.exists():
                rc = run(REF_PYTHON, "framework/bench_path_b.py", pilot, paths["overlay"], paths["provider"],
                         payload, paths["sdk"], "--scope", scope, "--events", "20000", "--pilot",
                         "--stock", stock, check=False)
                (output / f"B-{scope}-pilot.exit").write_text(f"{rc}\n")
            run(REF_PYTHON, "tools/benchmark_campaign.py", "recover-b-pilot", pilot, output / f"recovered-{scope}")
    elif stage == "measure-loading":
        run(REF_PYTHON, "packaging/compare_scoped_demo.py", "measure", paths["installation"],
            original, paths["partition"], scope, output / f"loading-{scope}",
            "--events", "20000", "--rounds", "4", "--verify-projection-identity")
    elif stage == "record-profile":
        run(python, "tools/profile_native.py", python, script, original, output / "native-profile", histogram,
            "--scope", scope, "--wheel", wheel, "--repeat", str(plan.get("profile_repeats", 10)),
            "--record-only", "--use-demo-defaults")
    elif stage == "time":
        if scope != "both":
            raise ValueError("The final timing grid schedules all three scopes itself")
        args = []
        if compact:
            proofs = output / "scoped-acceptance.json"
            proof = {s: {"b_recovery": f"recovered-{s}/recovery.json", "c_objects": f"C-{s}/comparison.json",
                         "c_shipped": f"shipped-{s}/comparison.json"} for s in compact}
            if proofs.exists():
                if json.loads(proofs.read_text()) != proof:
                    raise ValueError("Scoped proof manifest changed")
            else:
                with proofs.open("x") as stream:
                    json.dump(proof, stream, indent=2)
            for selected in compact:
                args.extend(["--compact-scope", selected])
            args.extend(["--payload-partition", paths["partition"], "--scoped-acceptance", proofs])
        if jobs is not None:
            args.extend(["--jobs", str(jobs)])
        run(REF_PYTHON, "tools/benchmark_campaign.py", "run", output / "production", "--cohort", "production",
            "--overlay", paths["overlay"], "--provider", paths["provider"], "--payload", paths["payload"],
            "--sdk", paths["sdk"], "--columnar-python", python, "--columnar-script", script,
            "--wheel", wheel, "--source-verification", REFERENCE / "full-clean-001/verification.json",
            "--b-verification", framework / "observer-comparison.json",
            "--c-verification", combined_c / "comparison.json", "--matched-read", *args)
    else:
        raise ValueError(f"Unknown qualification step {stage}")


if __name__ == "__main__":
    typer.run(main)
