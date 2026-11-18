"""Record guest competition around a command without changing any other process."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HZ = os.sysconf("SC_CLK_TCK")


def processes():
    found = {}
    for item in Path("/proc").iterdir():
        if not item.name.isdecimal():
            continue
        try:
            record = (item / "stat").read_text()
            end = record.rfind(")")
            fields = record[end + 2:].split()
            pid = int(item.name)
            identity = (pid, int(fields[19]))
            found[identity] = (int(fields[11]) + int(fields[12]), int(fields[36]), record[record.find("(")+1:end])
        except (FileNotFoundError, PermissionError, ProcessLookupError, IndexError, ValueError):
            continue
    return found


def project(pid):
    proc = Path("/proc") / str(pid)
    try:
        cwd = os.readlink(proc / "cwd")
        args = (proc / "cmdline").read_bytes().split(b"\0")
        paths = [cwd, *(os.fsdecode(arg) for arg in args)]
        if any(str(ROOT) in field for field in paths):
            return "jets"
        if any("muon-skill-" in field for field in paths):
            return "muon"
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return "exited"
    return "other"


def main():
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        raise SystemExit("Usage: watch_guest.py NEW.jsonl -- COMMAND [ARG ...]")
    path = Path(sys.argv[1]).resolve()
    os.sched_setaffinity(0, {5})
    previous = processes()
    begin = time.time_ns()
    with path.open("x", buffering=1) as out:
        out.write(json.dumps({"type":"start", "unix_ns":begin, "monitor_cpu":5,
                              "ticks_per_second":HZ, "command":sys.argv[3:]}) + "\n")
        child = subprocess.Popen(sys.argv[3:])
        while True:
            try:
                code = child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                code = None
            end = time.time_ns()
            current = processes()
            rows = []
            for identity, (ticks, cpu, name) in current.items():
                old = previous.get(identity)
                if old is None:
                    continue
                delta = (ticks - old[0]) / HZ
                if delta >= 0.05:
                    rows.append({"pid":identity[0], "program":name, "project":project(identity[0]),
                                 "last_cpu":cpu, "cpu_s":delta})
            cpu3 = next(line for line in Path("/proc/stat").read_text().splitlines() if line.startswith("cpu3 "))
            out.write(json.dumps({"type":"interval", "begin_unix_ns":begin, "end_unix_ns":end, "cpu3":cpu3,
                                  "io_pressure":Path("/proc/pressure/io").read_text().splitlines(),
                                  "active_processes":rows}) + "\n")
            if code is not None:
                out.write(json.dumps({"type":"end","unix_ns":end,"exit":code}) + "\n")
                return code
            begin, previous = end, current


if __name__ == "__main__":
    sys.exit(main())
