"""Prepare same-stream treatment/control variants from an established joint plan."""
import json
from pathlib import Path

import typer


def main(control: Path, candidate: Path, output: Path, rounds: int = 1, mechanism: str = "nominal-resolution-cache"):
    from hashlib import file_digest

    plan = json.loads(control.read_text())
    if plan["experiment"] != "rng-size" or rounds not in (1, 4):
        raise ValueError("Use the native baseline plan and one screening or four rotated rounds")
    choices = {"nominal-resolution-cache": "cache-nominal-resolution-tables", "nnjvt-inputs": "nnjvt-inputs"}
    if mechanism not in choices:
        raise ValueError(f"Unknown native comparison {mechanism}")
    expected = {"rng-cache", mechanism}
    if set((candidate.parent / "build-choices.txt").read_text().splitlines()) != expected:
        raise ValueError("The native trial must differ only by the chosen mechanism")
    manifest = (candidate.parent / "native.sha256").read_text()
    with candidate.open("rb") as source:
        if file_digest(source, "sha256").hexdigest() not in manifest:
            raise ValueError("Provider differs from its build receipt")
    on = dict(plan["variants"]["off"])
    on["provider"] = str(candidate.resolve())
    plan.update(experiment=choices[mechanism], rounds=rounds,
                variants={"off": plan["variants"]["off"], "on": on})
    plan["script_sources"] = {}
    for name in json.loads(control.read_text())["script_sources"]:
        with Path(name).open("rb") as stream:
            plan["script_sources"][name] = file_digest(stream, "sha256").hexdigest()
    if mechanism == "nnjvt-inputs":
        plan["scopes"] = ["small", "both"]
        for variant in plan["variants"].values():
            variant["scopes"] = plan["scopes"]
        plan["authority"] = {scope: plan["authority"][scope] for scope in plan["scopes"]}
    with output.open("x") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    typer.run(main)
