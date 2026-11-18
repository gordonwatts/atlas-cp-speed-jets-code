"""Compare the frozen faithful CLI and the installed packed and final CLIs."""
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from zipfile import ZipFile

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "ROOT_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ[variable] = "1"

import typer

from check_demo import CLEAN_ENV, digest
from compare_scoped_demo import ROOT, branch_digest, installed_commands, require_cpu

sys.path.insert(0, str(ROOT / "tools"))
from benchmark import CPU, EVENTS, REPETITIONS, check_quality, run_leg
from benchmark_campaign import faithful_startup_probe
from joint_trials import checksums
from profile_final_demo import checked_installation, histogram_comparison, stock_endpoint, validate_run
from profile_native import wheel_identity, write_json


SCOPES = ("small", "large", "both")
COLLECTIONS = {"small": ("sr",), "large": ("lr",), "both": ("sr", "lr")}
PAIR = ROOT / "results/benchmarks/paired-stream-001/pairs"
REFERENCE = ROOT / "results/reference"
PACKED = ROOT / "results/packaging/pristine-005"
PACKED_BUNDLE = ROOT / "results/packaging/bundle-005"
FAITHFUL_FILES = ("columnar/jets.py", "columnar/inputs.py", "jet_transport.py",
                  "config/jet-states.json", "config/jet-histogram.json", "tools/reference/setup.sh")
BASE_ENV = {**CLEAN_ENV, "ROOT_MAX_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def load(path):
    return json.loads(path.read_text())


def identity(path):
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def elf_receipts(output, members):
    output.mkdir()
    for path in members:
        with (output / f"{path.name}.txt").open("x") as log:
            subprocess.run(["readelf", "-dn", str(path)], env=BASE_ENV, stdout=log, check=True)


def installed(installation, bundle, output):
    commands, record = installed_commands(installation)
    python, demo = map(Path, commands["pip"])
    wheel = installation / "pip" / record["wheel"]
    metadata = bundle / "bundle.json"
    packaged, manifest = checked_installation(python, demo, wheel, metadata, digest, wheel_identity)
    if record.get("demo_defaults") != manifest["demo_defaults"]:
        raise ValueError(f"Transferred script defaults differ from the sealed bundle: {installation}")
    elf_receipts(output, [Path(member["path"]) for member in packaged["native"].values()])
    return {
        "python": python, "script": demo, "env": BASE_ENV,
        "packaged": packaged, "manifest": manifest,
        "provenance": {"installations": identity(installation / "installations.json"),
                       "wheel": packaged, "original_bundle": identity(metadata),
                       "visible_script": identity(demo), "native_elf": str(output),
                       "defaults": manifest["demo_defaults"]},
    }


def faithful(output, original, source, authorities):
    recorded = load(PAIR / "inputs.json")
    plan = recorded["plan"]
    arm = plan["variants"]["faithful"]
    revision = recorded["source_snapshot"]
    if (not re.fullmatch(r"[0-9a-f]{40}", revision) or plan["events"] != EVENTS or arm["options"]
            or set(arm["scopes"]) != set(SCOPES) or Path(plan["input"]).resolve() != original):
        raise ValueError("The archived faithful plan does not cover the frozen full input")
    paths = {}
    for name in FAITHFUL_FILES:
        data = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{revision}:{name}"])
        paths[name] = data
    if sha256(paths["columnar/jets.py"]).hexdigest() != recorded["source_sha256"]:
        raise ValueError("The committed faithful CLI does not match the checked paired-stream source")
    provider = Path(arm["provider"]).resolve(strict=True)
    python = Path(plan["python"])
    if not python.is_file():
        raise FileNotFoundError(python)
    provenance = Path(plan["provenance"]).resolve(strict=True)
    precision = python.parents[2]
    core = python.parents[1] / "lib/python3.14/site-packages/correctionlib/lib/libcorrectionlib.so"
    saved_digests = recorded["digests"]["faithful"]
    build_inputs = checksums(provider.parent / "build-inputs.sha256")
    if digest(provider) != saved_digests["provider"] or digest(core) != build_inputs.get(core):
        raise ValueError("The original faithful provider or evaluator has changed")
    (wheel, expected_wheel), = checksums(precision / "wheel.sha256").items()
    if digest(wheel) != expected_wheel:
        raise ValueError("The original precision evaluator wheel has changed")
    site = core.parents[2]
    with ZipFile(wheel) as archive:
        members = [name for name in archive.namelist() if name == "correctionlib/lib/libcorrectionlib.so"
                   or re.fullmatch(r"correctionlib/_core\.cpython-314-[^/]+\.so", name)]
        if len(members) != 2 or any(digest(site / name) != sha256(archive.read(name)).hexdigest() for name in members):
            raise ValueError("Installed faithful evaluator members differ from the precision wheel")
    binding, = (site / name for name in members if "/_core." in name)

    receipts = {}
    payloads = {}
    for scope in SCOPES:
        work = PAIR / f"01-{scope}-faithful"
        row = load(work / "receipt.json")
        payload = Path(arm["payloads"][scope])
        if digest(payload) != saved_digests["payloads"][scope]:
            raise ValueError(f"Faithful paired-stream payload changed for {scope}")
        stock = authorities[scope]["histogram"]
        if Path(plan["authority"][scope]).resolve() != Path(stock["path"]).resolve():
            raise ValueError(f"Faithful paired-stream stock histogram changed for {scope}")
        argv = [str(python), plan["script"], plan["input"], str(work / "histogram.npz"),
                "--provider", arm["provider"], "--payload", str(payload), "--provenance", plan["provenance"],
                "--scope", scope, "--mode", "full", "--events", str(EVENTS)]
        if row["command"] != argv or row["exit"] or not row["process"]["passed"] or not row["exact_histogram"]:
            raise ValueError(f"The checked faithful invocation cannot be recovered for {scope}")
        log = work / "run.log"
        if f"({source['tfile_uuid']})" not in log.read_text():
            raise ValueError(f"Faithful original UUID differs for {scope}")
        receipts[scope] = {"receipt": identity(work / "receipt.json"), "log": identity(log),
                           "branch_sha256": branch_digest(log)}
        payloads[scope] = identity(payload)

    link_receipt = provider.parent / "link-libraries.txt"
    wanted = {"libJetCalibToolsLib.so", "libJetUncertaintiesLib.so", "libCore.so"}
    mapped = {name: Path(path).resolve(strict=True) for name, path in re.findall(
        r"^\s*(\S+)\s+=>\s+(/cvmfs/\S+)\s+", link_receipt.read_text(), re.MULTILINE) if name in wanted}
    if mapped.keys() != wanted:
        raise ValueError("The original faithful ATLAS/ROOT linkage receipt is unavailable")

    for name, content in paths.items():
        copied = output / name
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes(content)
    setup = output / "tools/reference/setup.sh"
    setup_log = output / "release-setup.log"
    inherited = {**BASE_ENV, **{key: os.environ[key] for key in ("HOME", "USER", "LOGNAME") if key in os.environ}}
    with setup_log.open("x") as log:
        block = subprocess.check_output(
            ["bash", "-c", 'source "$1" >&2 && env -0', "faithful-release", str(setup)],
            env=inherited, stderr=log, cwd=output,
        )
    release = dict(item.split("=", 1) for item in os.fsdecode(block).rstrip("\0").split("\0"))
    if (release.get("AtlasVersion") != "25.2.117" or release.get("CMTCONFIG") != "x86_64-el9-gcc15-opt"
            or Path(release["JETS_REFERENCE_INPUT"]).resolve() != original):
        raise ValueError("Frozen release setup failed to select the original release and PHYSLITE")
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD"):
        release.pop(key, None)
    release["LD_LIBRARY_PATH"] = f"{core.parent}:{release['LD_LIBRARY_PATH']}"
    release.update({key: value for key, value in BASE_ENV.items() if key.endswith("_THREADS")})
    elf = output / "elf"
    elf_receipts(elf, [provider, core, binding])
    script = output / "columnar/jets.py"
    return {
        "python": python, "script": script, "env": release, "provider": provider, "core": core,
        "binding": binding, "mapped": mapped, "payloads": payloads, "payload_provenance": provenance,
        "historical": receipts,
        "provenance": {
            "paired_stream_input_receipt": identity(PAIR / "inputs.json"), "paired_stream_scopes": receipts,
            "source_revision": revision, "source_files": {name: identity(output / name) for name in paths},
            "original_precision_wheel": identity(wheel), "provider": identity(provider),
            "evaluator": identity(core), "binding": identity(binding), "provider_linkage": identity(link_receipt),
            "release_libraries": {key: identity(path) for key, path in mapped.items()}, "elf": str(elf),
            "payloads": payloads, "payload_provenance": identity(provenance),
            "environment_policy": "Recreate the pinned AnalysisBase setup and load the original precision core. "
                                  "The inherited historical environment was not saved by the paired-stream controller.",
            "launch_environment": {key: release.get(key) for key in ("AtlasVersion", "AtlasArea", "CMTCONFIG",
                                "PATH", "ROOTSYS", "ROOT_INCLUDE_PATH", "LD_LIBRARY_PATH", "JETS_REFERENCE_INPUT")},
            "defaults": {"mode": "nominal", "scope": "both", "repeat": 1, "effective_mode": "full",
                         "effective_options": arm["options"]},
        },
    }


def schedule(names):
    for index in range(REPETITIONS):
        scopes = SCOPES[index % len(SCOPES):] + SCOPES[:index % len(SCOPES)]
        for scope in scopes:
            offset = (index + SCOPES.index(scope)) % len(names)
            for name in names[offset:] + names[:offset]:
                yield index + 1, scope, name


def execute(name, arm, scope, original, source, objects, stock, work):
    output = work / "histograms.npz"
    cli = [str(arm["python"]), str(arm["script"]), str(original), str(output)]
    env = arm["env"]
    startup_validator = None
    if name == "faithful":
        cli += ["--provider", str(arm["provider"]), "--payload", arm["payloads"][scope]["path"],
                "--provenance", str(arm["payload_provenance"]), "--scope", scope, "--mode", "full"]
        def startup_validator(child, family):
            return faithful_startup_probe(child, family, arm["python"].resolve())
    elif scope != "both":
        cli.extend(("--scope", scope))
    cli.extend(("--events", str(EVENTS)))
    process = run_leg(cli, env, work, "visible", event_source=original, startup_validator=startup_validator)
    if not process["source_mapping_observed"]:
        raise ValueError(f"The input file mapping was not observed: {work}")
    log = work / "visible.log"
    branches = branch_digest(log)
    if name == "faithful":
        required = {arm["provider"], arm["core"].resolve(), arm["binding"].resolve(), *arm["mapped"].values()}
        observed = {Path(path).resolve() for path in process["libraries"]}
        if not required <= observed:
            raise ValueError(f"Faithful runtime maps missed original libraries: {sorted(required - observed)}")
        text = log.read_text()
        if (branches != arm["historical"][scope]["branch_sha256"]
                or text.count(f"({source['tfile_uuid']})") != 1
                or text.count(f"{EVENTS} unique events, {EVENTS} event visits across 1 passes") != 1):
            raise ValueError(f"Faithful input stream differs from the archived invocation: {work}")
        measured = {prefix: int(count) for prefix, count in re.findall(r"(?m)^(sr|lr) (\d+) unique jets,", text)}
        details = {"objects": measured, "branch_sha256": branches}
    else:
        # Reuse the installed identity gate with the collector's maps. The
        # benchmark CLI runs without the diagnostic library-audit environment.
        write_json(work / "library-maps.json", {"libraries": process["libraries"]})
        details = validate_run(work, scope, EVENTS, source["tfile_uuid"], arm["manifest"], arm["packaged"])
    if details["objects"] != {prefix: objects[prefix] for prefix in COLLECTIONS[scope]}:
        raise ValueError(f"Original jet visits differ for {scope}: {work}")
    stock_file = Path(stock["path"])
    if digest(stock_file) != stock["sha256"]:
        raise ValueError(f"Independent stock endpoint changed: {stock_file}")
    exact = histogram_comparison(stock_file, output, scope, EVENTS)
    if not all(exact.values()):
        raise ValueError(f"Histogram fields are not bitwise exact: {work}")
    record = {"execution": str(work / "visible.json"), "wall_s": process["wall_s"],
              "cpu_s": process["cpu_user_s"] + process["cpu_sys_s"],
              "exact_stock_fields": exact, **details}
    write_json(work / "comparison.json", record)
    return record


def summarize(evidence):
    results = {}
    for scope, arms in evidence.items():
        stats = {}
        for name, rows in arms.items():
            cpu = [row["cpu_s"] for row in rows[1:]]
            stats[name] = {"wall": check_quality(rows), "retained_cpu_s": cpu,
                           "mean_cpu_s": statistics.mean(cpu), "sample_sd_cpu_s": statistics.stdev(cpu)}
        ratios = {}
        for left, right in (("faithful", "packed"), ("packed", "final"), ("faithful", "final")):
            if {left, right} <= stats.keys():
                a, b = stats[left]["wall"], stats[right]["wall"]
                ratios[f"{left}_over_{right}"] = {
                    "ratio_of_retained_means": a["mean_s"] / b["mean_s"],
                    "same_round_wall_ratios": [x / y for x, y in zip(a["retained"], b["retained"], strict=True)],
                    "qualified_for_spread": a["pass"] and b["pass"],
                }
        results[scope] = {"arms": stats, "speedup_ratios": ratios}
    return results


def main(final_installation: Path, final_bundle: Path, output: Path, prepare_only: bool = False):
    """Run the optional installed optimization progression on the original 20,000-event file."""
    final_installation, final_bundle = (path.resolve(strict=True) for path in (final_installation, final_bundle))
    output = output.resolve()
    if not prepare_only:
        require_cpu()
    source = load(ROOT / "config/jets-input.json")
    original = Path(source["path"]).resolve(strict=True)
    independent = load(REFERENCE / "full-clean-001/verification.json")
    if (digest(original) != source["sha256"] or independent["source_file_uuid"] != source["tfile_uuid"]
            or independent["events"] != EVENTS):
        raise ValueError("The independent observer and frozen input do not identify the original full file")
    authorities = {scope: stock_endpoint(REFERENCE / f"full-file-pilot-{('combined' if scope == 'both' else scope)}-002",
                                       scope, original, source, digest) for scope in SCOPES}
    output.mkdir()
    arms = {}
    omission = None
    try:
        arms["faithful"] = faithful(output / "faithful-source", original, source, authorities)
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as error:
        omission = str(error)
        typer.echo(f"Faithful baseline omitted because the archived runtime could not be recovered. {omission}", err=True)
    arms["packed"] = installed(PACKED, PACKED_BUNDLE, output / "packed-elf")
    arms["final"] = installed(final_installation, final_bundle, output / "final-elf")
    jobs = list(schedule(tuple(arms)))
    write_json(output / "provenance.json", {
        "input": source, "authority": authorities, "events": EVENTS, "scopes": SCOPES, "cpu": CPU, "threads": 1,
        "controller": identity(Path(__file__).resolve()), "rounds": REPETITIONS,
        "policy": "Fresh CLI per cell. Rotate scope and arm order. Exclude round 1. Warm OS cache is allowed. "
                  "Wall time includes startup and saved histogram. CPU includes the observed worker process.",
        "launch_environment": BASE_ENV, "faithful_omission": omission, "schedule": jobs,
        "arms": {name: arm["provenance"] for name, arm in arms.items()},
    })
    if prepare_only:
        return
    evidence = {scope: {name: [] for name in arms} for scope in SCOPES}
    branches = {}
    with (ROOT / f"results/.benchmark-cpu{CPU}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for round_no, scope, name in jobs:
            work = output / f"{round_no:02d}-{scope}-{name}"
            work.mkdir()
            row = execute(name, arms[name], scope, original, source,
                          {key: value["objects"] for key, value in independent["collections"].items()},
                          authorities[scope]["histogram"], work)
            branch = row["branch_sha256"]
            if branches.setdefault(scope, branch) != branch:
                raise ValueError(f"Physical branch requests disagree for {scope}: {work}")
            evidence[scope][name].append(row)
    write_json(output / "summary.json", {
        "all_histograms_exact": True, "matching_physical_branch_sha256": branches,
        "faithful_omission": omission, "results": summarize(evidence),
    })


if __name__ == "__main__":
    typer.run(main)
