"""Prepare same-stream treatment/control variants from an established joint plan."""
import json
from pathlib import Path

import typer


def main(control: Path, candidate: Path, output: Path, rounds: int = 1):
    from hashlib import file_digest

    plan = json.loads(control.read_text())
    if plan["experiment"] != "rng-size" or rounds not in (1, 4):
        raise ValueError("Use the native baseline plan and one screening or four rotated rounds")
    expected = {"rng-cache", "nominal-resolution-cache"}
    if set((candidate.parent / "build-choices.txt").read_text().splitlines()) != expected:
        raise ValueError("The nominal-table trial must differ only by that mechanism")
    manifest = (candidate.parent / "native.sha256").read_text()
    with candidate.open("rb") as source:
        if file_digest(source, "sha256").hexdigest() not in manifest:
            raise ValueError("Provider differs from its build receipt")
    on = dict(plan["variants"]["off"])
    on["provider"] = str(candidate.resolve())
    plan.update(experiment="cache-nominal-resolution-tables", rounds=rounds,
                variants={"off": plan["variants"]["off"], "on": on})
    with output.open("x") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    typer.run(main)
