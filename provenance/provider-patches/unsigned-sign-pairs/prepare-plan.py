"""Create a same-provider off/on plan for the sign-pair integration patch."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

OPTION = "--reuse-uncertainty-sign-pairs"


def main(template, provider, output, rounds):
    if rounds < 1:
        raise ValueError("Rounds must be positive")
    plan = json.loads(template.read_text())
    matched = deepcopy(plan["variants"]["off"])
    if OPTION in matched.get("options", ()):
        raise ValueError("Template control already enables sign-pair reuse")
    matched["provider"] = str(provider.resolve(strict=True))
    candidate = deepcopy(matched)
    candidate["options"] = [*matched.get("options", ()), OPTION]
    plan["variants"] = {"off": matched, "on": candidate}
    plan["experiment"] = "unsigned-sign-pairs"
    plan["rounds"] = rounds
    plan["script_sources"] = {p: sha256(Path(p).read_bytes()).hexdigest() for p in plan["script_sources"]}
    with output.open("x") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("usage: prepare-plan.py MATCHED-JOINT-TEMPLATE PROVIDER NEW-PLAN ROUNDS")
    main(*map(Path, sys.argv[1:4]), int(sys.argv[4]))
