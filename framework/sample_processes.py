"""Sample the framework job and spawned processes without modifying their environment."""
import json
from pathlib import Path
import sys
import time


def sample(pid, output):
    root = Path("/proc")
    process = root / str(pid)
    while not output.parent.exists() and process.exists():
        time.sleep(0.05)
    with output.open("w") as stream:
        while process.exists():
            queue = [pid]
            rows = []
            while queue:
                current = queue.pop()
                proc = root / str(current)
                try:
                    status = dict(
                        line.split(":", 1) for line in (proc / "status").read_text().splitlines()
                        if ":" in line
                    )
                    command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
                    children = (proc / "task" / str(current) / "children").read_text().split()
                except (FileNotFoundError, ProcessLookupError):
                    continue
                rows.append({
                    "pid": current, "exec": command, "threads": int(status["Threads"]),
                    "allowed_cpus": status["Cpus_allowed_list"].strip(),
                })
                queue.extend(map(int, children))
            stream.write(json.dumps({"monotonic": time.monotonic(), "processes": rows}) + "\n")
            stream.flush()
            time.sleep(0.20)


if __name__ == "__main__":
    sample(int(sys.argv[1]), Path(sys.argv[2]))
