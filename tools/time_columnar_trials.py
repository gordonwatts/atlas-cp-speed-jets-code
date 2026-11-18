"""Run rotated, same-stream full-file engineering trials with fixed direct histograms."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import select
import subprocess
import time

import numpy as np
import typer


def hashed(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def variant_runtime(plan, variant, scope):
    python = variant.get("python", plan.get("python"))
    sdk = variant.get("sdk", plan.get("sdk"))
    payload = variant["payloads"][scope]
    provenance = variant.get("provenances", {}).get(scope, variant.get("provenance", plan.get("provenance")))
    if not python or not provenance:
        raise ValueError(f"Trial {scope} requires a Python and explicit payload provenance")
    return {"python": python, "sdk": sdk, "payload": payload, "provenance": provenance}


def runtime_environment(base, sdk):
    env = base.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    if sdk:
        compiler = env.get("LD_LIBRARY_PATH")
        search = [str(Path(sdk) / "lib")]
        if compiler:
            search.append(compiler)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(search)
    return env


def verify_script_sources(plan):
    for filename, digest in plan.get("script_sources", {}).items():
        if hashed(filename) != digest:
            raise ValueError(f"Trial source drifted after plan generation: {filename}")


def source_branch_digest(log):
    prefix = "Input branch SHA-256: "
    candidates = [line[len(prefix):].strip() for line in log.read_text().splitlines() if line.startswith(prefix)]
    if len(candidates) != 1 or len(candidates[0]) != 64 or any(ch not in "0123456789abcdef" for ch in candidates[0]):
        raise ValueError(f"Missing or malformed original branch digest in {log}")
    return candidates[0]


def snapshot():
    cpus = [line for line in Path("/proc/stat").read_text().splitlines() if line.startswith("cpu")]
    return {"cpu3":next(line for line in cpus if line.startswith("cpu3 ")), "all_guest_cpus":cpus,
            "io_pressure":Path("/proc/pressure/io").read_text(),
            "top":subprocess.check_output(["ps","-eo","pid,psr,pcpu,comm","--sort=-pcpu"],text=True).splitlines()[:12]}


def wait_for_child(pid, expected_libraries):
    seen = set()
    samples = 0
    libraries = {name: set() for name in expected_libraries}
    fd = os.pidfd_open(pid)
    try:
        while True:
            try:
                values = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines()
                              if ":" in line)
                seen.add((int(values["Threads"]), values["Cpus_allowed_list"].strip()))
                samples += 1
                if expected_libraries and not all(libraries.values()):
                    with Path(f"/proc/{pid}/maps").open() as mapped:
                        for line in mapped:
                            fields = line.split(None, 5)
                            if len(fields) == 6:
                                candidate = Path(fields[5].strip())
                                if candidate.name in libraries:
                                    libraries[candidate.name].add(str(candidate.resolve()))
            except (FileNotFoundError, ProcessLookupError):
                pass
            ready, _, _ = select.select([fd], [], [], 0.5)
            if ready:
                break
        _, status, usage = os.wait4(pid, 0)
    finally:
        os.close(fd)
    loaded = {name: sorted(paths) for name, paths in libraries.items()}
    library_match = all(libraries[name] == {str(Path(path).resolve())} for name, path in expected_libraries.items())
    return status, usage, {"samples": samples, "states": sorted(seen), "loaded_libraries": loaded,
                           "passed": seen == {(1, "3")} and library_match}

def save(path,value):
    with path.open("x") as stream:
        json.dump(value,stream,indent=2)
        stream.write("\n")


def main(plan:Path, output:Path):
    if os.sched_getaffinity(0)!={3} or len(list(Path("/proc/self/task").iterdir()))!=1:
        raise ValueError("Run on the single reserved analysis CPU/thread")
    config=json.loads(plan.read_text())
    root=Path(__file__).resolve().parents[1]
    verify_script_sources(config)
    output.mkdir(exist_ok=False)
    variants=config["variants"]
    names=list(variants)
    scopes=config["scopes"]
    rounds=config.get("rounds",4)
    events=config["events"]
    if len(names)<2 or rounds<1 or events!=20000:
        raise ValueError("Full-file candidate comparisons need at least two distinct configurations")
    env=dict(os.environ)
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","ROOT_MAX_THREADS","BLIS_NUM_THREADS"):
        env[key]="1"
    env.pop("PYTHONHOME",None)
    env.pop("PYTHONPATH",None)
    receipts={}
    for name,variant in variants.items():
        active = variant.get("scopes", scopes)
        if not set(active).issubset(scopes) or not active:
            raise ValueError(f"Invalid scope subset for {name}")
        args = variant.get("options", [])
        if any(arg not in {"--reuse-nominal-nnjvt-update", "--reuse-uncertainty-sign-pairs"} for arg in args):
            raise ValueError(f"Unsupported trial options for {name}")
        runtime = {scope: variant_runtime(config, variant, scope) for scope in active}
        sdk = runtime[active[0]]["sdk"]
        receipts[name] = {
            "python": runtime[active[0]]["python"],
            "core": hashed(Path(sdk) / "lib/libcorrectionlib.so") if sdk else None,
            "provider": hashed(variant["provider"]),
            "payloads": {scope: hashed(spec["payload"]) for scope, spec in runtime.items()},
            "provenance": {scope: hashed(spec["provenance"]) for scope, spec in runtime.items()},
        }
    save(output/"inputs.json",{"plan":config,"digests":receipts,"source_sha256":hashed(config["script"]),
                              "source_snapshot":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
                              "starts":snapshot(),"policy":"Four rounds, first excluded. A one-round diagnostic gives no retained mean."})
    jobs=[]
    branch_digests={}
    for r in range(rounds):
        order=scopes[r%len(scopes):]+scopes[:r%len(scopes)]
        for i,scope in enumerate(order):
            available = [name for name in names if scope in variants[name].get("scopes", scopes)]
            if len(available) < 2:
                raise ValueError(f"Too few trials for {scope}")
            offset=(r+i)%len(available)
            visit=available[offset:]+available[:offset]
            for name in visit:
                v=variants[name]
                job=output/f"{r+1:02d}-{scope}-{name}"
                job.mkdir()
                result=job/"histogram.npz"
                runtime = variant_runtime(config, v, scope)
                sdk = runtime["sdk"]
                job_env = runtime_environment(env, sdk)
                expected = {"libjets.so": v["provider"], "libcorrectionlib.so": Path(sdk) / "lib/libcorrectionlib.so"} if sdk else {}
                command=[runtime["python"],config["script"],config["input"],str(result.resolve()),
                         "--provider",v["provider"],"--payload",runtime["payload"],
                         "--provenance",runtime["provenance"],"--scope",scope,"--mode","full","--events",str(events)]
                command.extend(v.get("options", []))
                verify_script_sources(config)
                before=snapshot()
                with (job/"run.log").open("x") as log:
                    start=time.perf_counter()
                    proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=job_env,cwd=root)
                    status,usage,process=wait_for_child(proc.pid,expected)
                    elapsed=time.perf_counter()-start
                    proc.returncode=os.waitstatus_to_exitcode(status)
                record={"name":name,"scope":scope,"round":r+1,"command":command,"exit":proc.returncode,
                        "wall_s":elapsed,"cpu_s":usage.ru_utime+usage.ru_stime,"peak_rss_kB":usage.ru_maxrss,
                        "before":before,"after":snapshot(),"process":process}
                if proc.returncode==0:
                    with np.load(config["authority"][scope],allow_pickle=False) as a,np.load(result,allow_pickle=False) as c:
                        record["exact_histogram"]=set(a.files)==set(c.files) and all(np.array_equal(c[k],a[k]) for k in a.files)
                if proc.returncode == 0 and config.get("require_matching_branches"):
                    try:
                        digest = source_branch_digest(job / "run.log")
                        record["source_branch_sha256"] = digest
                        record["matching_source_branches"] = branch_digests.setdefault(scope, digest) == digest
                    except ValueError as error:
                        record.update(matching_source_branches=False, branch_error=str(error))
                save(job/"receipt.json",record)
                if proc.returncode or not record["exact_histogram"] or not process["passed"] or not record.get("matching_source_branches", True):
                    raise RuntimeError(f"Unsuccessful or unequal trial {job}")
                jobs.append(record)
    metrics={}
    if rounds>=4:
        for scope in scopes:
            metrics[scope]={}
            for name in names:
                samples=[j["wall_s"] for j in jobs if j["scope"]==scope and j["name"]==name and j["round"]>1]
                if not samples:
                    continue
                mean,median=statistics.mean(samples),statistics.median(samples)
                sd=statistics.stdev(samples)
                metrics[scope][name]={"wall_s":samples,"mean":mean,"sample_sd":sd,
                                      "within_frozen_spread":sd/mean<=0.05 and max(abs(s-median)/median for s in samples)<=0.1}
    save(output/"summary.json",{"observations":len(jobs),"metrics":metrics,"all_histograms_equal":True,
                                "resource_review_required":True})

if __name__=="__main__":
    typer.run(main)
