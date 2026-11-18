"""Install the transferred bundle cleanly and verify its visible analysis command."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import typer

APP = typer.Typer(add_completion=False)
CLEAN_ENV = {
    "PATH": "/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "PYTHONNOUSERSITE": "1",
    "UV_CONCURRENT_BUILDS": "1",
    "UV_CONCURRENT_INSTALLS": "1",
    "UV_CONCURRENT_DOWNLOADS": "1",
    **{name: "1" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
        "RAYON_NUM_THREADS",
    )},
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(command, cwd: Path, label: str, cpu: int, additions=None, expect_success=True):
    env = dict(CLEAN_ENV)
    env.update(additions or {})
    command = ["taskset", "-c", str(cpu), *map(str, command)]
    with (cwd / f"{label}.argv.json").open("x") as stream:
        json.dump(command, stream, indent=2)
        stream.write("\n")
    with (cwd / f"{label}.log").open("xb") as log:
        outcome = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    (cwd / f"{label}.exit").write_text(f"{outcome.returncode}\n")
    if (outcome.returncode == 0) != expect_success:
        raise RuntimeError(f"Unexpected {label} exit {outcome.returncode}; see {cwd}")
    return outcome.returncode


def script_dependencies(path: Path):
    _, header, _ = path.read_text().split("# ///", 2)
    lines = header.splitlines()[1:]
    config = tomllib.loads("\n".join(line.removeprefix("# ").removeprefix("#") for line in lines))
    return [name for name in config["dependencies"] if not name.startswith("correctionlib==")]


@APP.command()
def prepare(bundle: Path, result: Path, python: Path, uv: Path) -> None:
    """Install fresh pip and PEP 723 environments without executing event analysis."""
    bundle, result, python, uv = (path.resolve() for path in (bundle, result, python, uv))
    result.mkdir()
    manifest = json.loads((bundle / "bundle.json").read_text())
    wheel_name = manifest["wheel"]["name"]
    script_name = manifest["visible_script"]["name"]
    for name in ("pip", "uv"):
        target = result / name
        target.mkdir()
        for asset, key in ((wheel_name, "wheel"), (script_name, "visible_script")):
            shutil.copyfile(bundle / asset, target / asset)
            if digest(target / asset) != manifest[key]["sha256"]:
                raise ValueError(f"Transferred file does not match the frozen bundle: {asset}")

    pip = result / "pip"
    pip_python = pip / "env/bin/python"
    run([python, "-VV"], pip, "interpreter", 5)
    run([python, "-m", "venv", pip / "env"], pip, "create-env", 5)
    run([pip_python, "-m", "pip", "--isolated", "install", "--no-cache-dir",
         pip / wheel_name, *script_dependencies(pip / script_name)], pip, "install", 5)
    run([pip_python, "-m", "pip", "check"], pip, "dependency-check", 5)
    run([pip_python, "-m", "pip", "freeze", "--all"], pip, "packages", 5)

    uv_dir = result / "uv"
    cache = uv_dir / "cache"
    run([uv, "--version"], uv_dir, "uv-version", 5)
    run([uv, "sync", "--no-config", "--native-tls", "--no-python-downloads",
         "--python", python, "--cache-dir", cache, "--script", uv_dir / script_name],
        uv_dir, "install", 5)
    record = {"python": str(python), "uv": str(uv), "script": script_name, "wheel": wheel_name,
              "wheel_sha256": manifest["wheel"]["sha256"], "script_sha256": manifest["visible_script"]["sha256"]}
    (result / "installations.json").write_text(json.dumps(record, indent=2) + "\n")


@APP.command()
def smoke(result: Path, original: Path, reference_nominal: Path, events: int = 32,
          matched_read: bool = False) -> None:
    """Run the same visible script via pristine pip and uv, and check the independent nominal control."""
    result = result.resolve()
    original = original.resolve(strict=True)
    reference_nominal = reference_nominal.resolve(strict=True)
    record = json.loads((result / "installations.json").read_text())
    commands = {
        "pip": [result / "pip/env/bin/python", result / "pip" / record["script"]],
        "uv": [record["uv"], "run", "--no-project", "--no-config", "--native-tls",
               "--no-python-downloads", "--python", record["python"], "--cache-dir",
               result / "uv/cache", result / "uv" / record["script"]],
    }
    extras = ["--matched-read"] if matched_read else []
    for name, command in commands.items():
        cwd = result / name
        run([*command, original, cwd / "full.npz", "--events", events, "-n", 2, *extras],
            cwd, "full", 3, {"ATLAS_JET_LIBRARY_AUDIT": str(cwd / "full-library-maps.json")})
    pip = result / "pip"
    run([*commands["pip"], original, pip / "nominal.npz", "--events", events, "--mode", "nominal", *extras],
        pip, "nominal", 3, {"ATLAS_JET_LIBRARY_AUDIT": str(pip / "nominal-library-maps.json")})
    if matched_read:
        label = "Input branch SHA-256:"
        digests = {}
        for name, mode in (("pip", "full"), ("uv", "full"), ("pip", "nominal")):
            lines = (result / name / f"{mode}.log").read_text().splitlines()
            value, = (line.split()[-1] for line in lines if line.startswith(label))
            digests[f"{name}_{mode}"] = value
        if len(set(digests.values())) != 1:
            raise RuntimeError(f"Matched-read run resolved different branches: {digests}")
        (result / "branch-digests.json").write_text(json.dumps(digests, indent=2) + "\n")
    for suffix, count in (("zero", "0"), ("negative", "-1"), ("noninteger", "1.5")):
        run([*commands["pip"], original, pip / f"bad-{suffix}.npz", "--events", events, "-n", count],
            pip, f"bad-{suffix}", 3, expect_success=False)
        if (pip / f"bad-{suffix}.npz").exists():
            raise RuntimeError(f"Invalid repeat {count} wrote an output histogram")

    # Use the same isolated Python for parsing the evidence and reference.
    run([pip / "env/bin/python", Path(__file__).resolve(), "_compare", result, reference_nominal],
        pip, "comparison", 3)


def compare(result: Path, reference_nominal: Path) -> None:
    import numpy as np

    with np.load(result / "pip/full.npz") as pip, np.load(result / "uv/full.npz") as uv:
        if pip.files != uv.files:
            raise ValueError("The two installed script outputs have different fields")
        for field in pip.files:
            np.testing.assert_array_equal(pip[field], uv[field], err_msg=field)
        states = {scope: len(pip[f"{scope}_states"]) for scope in ("sr", "lr")}
    with np.load(result / "pip/nominal.npz") as one, np.load(result / "pip/full.npz") as many:
        for name in one.files:
            expected = many[name] if name in ("edges", "events") else many[name][..., :1]
            np.testing.assert_array_equal(one[name], expected, err_msg=f"matched NOSYS {name}")
    errors = {}
    with np.load(result / "pip/nominal.npz") as candidate, np.load(reference_nominal) as authority:
        for field in authority.files:
            got, expected = candidate[field], authority[field]
            if field.endswith(("sumw", "sumw2")):
                errors[field] = float(np.max(np.abs(got - expected)))
                np.testing.assert_allclose(got, expected, rtol=2e-6, atol=2e-6, err_msg=field)
            else:
                np.testing.assert_array_equal(got, expected, err_msg=field)
    (result / "smoke-comparison.json").write_text(json.dumps({
        "pip_uv_full_exact": True,
        "pip_nominal_full_exact": True,
        "original_nominal_control": str(reference_nominal),
        "nominal_histogram_max_absolute_differences": errors,
        "state_counts": states,
    }, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "_compare":
        compare(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        APP()
