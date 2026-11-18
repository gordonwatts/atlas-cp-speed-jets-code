"""Exercise real proc-map sampling with short workers and no event input."""

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import typer

from benchmark import CPU, run_leg


def main(library_dir: Path) -> None:
    """Dynamically load the two installed compiler libraries in a short worker."""
    if os.sched_getaffinity(0) != {CPU}:
        raise typer.BadParameter(f"Pin the fixture to CPU {CPU}")

    library_dir = library_dir.resolve(strict=True)
    libraries = tuple(str((library_dir / name).resolve(strict=True)) for name in
                      ("libgcc_s.so.1", "libstdc++.so.6"))
    code = (
        "import ctypes, sys, time\n"
        "time.sleep(0.04)\n"
        "handles = [ctypes.CDLL(filename) for filename in sys.argv[1:]]\n"
        "time.sleep(0.27)\n"
    )
    command = [sys.executable, "-S", "-c", code, *libraries]
    env = {**os.environ, "LD_LIBRARY_PATH": str(library_dir)}
    env.pop("LD_PRELOAD", None)
    with TemporaryDirectory() as scratch:
        directory = Path(scratch)
        regular = run_leg(command, env, directory, "regular")
        startup = run_leg(command, env, directory, "startup", capture_startup_libraries=True)

    # This reproduces an import-and-exit worker ending before the 500 ms read.
    assert regular["wall_s"] < 0.5 and startup["wall_s"] < 0.5, "Fixture ran too long to test startup cadence"
    assert not set(libraries).issubset(regular["libraries"]), "Regular map cadence unexpectedly saw both libraries"
    assert set(libraries).issubset(startup["libraries"]), "Startup map sampling missed an installed library"
    typer.echo(f"Regular cadence missed a mapping at {regular['wall_s']:.3f} s; "
               f"startup sampling found both at {startup['wall_s']:.3f} s.")


if __name__ == "__main__":
    typer.run(main)
