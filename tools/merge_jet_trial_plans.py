"""Rotate accepted single-change configurations together with shared controls."""
import copy
from hashlib import file_digest
import json
from pathlib import Path
from typing import Annotated

import typer


def digest(path):
    with Path(path).open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def row_for(variant, scope):
    row = dict(variant)
    row.pop("scopes")
    row["payloads"] = row["payloads"][scope]
    row["provenances"] = row["provenances"][scope]
    return row


def main(plans: Path, output: Path,
         family: Annotated[list[str], typer.Option(help="Joint plan family to repeat")],
         extra: Annotated[list[Path] | None, typer.Option(help="Additional standalone pair plan")] = None,
         rounds: Annotated[int, typer.Option(min=4)] = 4):
    base_file = plans / "rng-size.json"
    common = json.loads(base_file.read_text())
    configurations = {"control": common["variants"]["off"]}
    comparisons = {}
    records = {}
    files = [*(plans / f"{name}.json" for name in family), *(extra or ())]
    for filename in files:
        plan = json.loads(filename.read_text())
        if any(plan[key] != common[key] for key in ("input", "script", "events", "observer")):
            raise ValueError(f"Original sample, entry point, or observer differs in {filename}")
        if any(plan["authority"][scope] != common["authority"][scope] for scope in plan["scopes"]):
            raise ValueError(f"Stock histogram differs in {filename}")
        label = plan["experiment"]
        if label in comparisons or "/" in label:
            raise ValueError(f"Duplicate or unsafe trial name {label}")
        off = plan["variants"]["off"]
        scope = off["scopes"]
        matches = [key for key, variant in configurations.items()
                   if set(scope) <= set(variant["scopes"])
                   and all(row_for(off, item) == row_for(variant, item) for item in scope)]
        control = matches[0] if matches else f"{label}-off"
        if not matches:
            configurations[control] = off
        configurations[label] = plan["variants"]["on"]
        comparisons[label] = {"control": control, "scopes": scope}
        records[str(filename.resolve())] = digest(filename)
    result = copy.deepcopy(common)
    result.update(experiment="rotated-independent-changes", rounds=rounds, variants=configurations,
                  comparisons=comparisons, source_plans=records)
    # Freeze the current visible source once for all arms after opt-in additions.
    result["script_sources"] = {name: digest(name) for name in result["script_sources"]}
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    typer.run(main)
