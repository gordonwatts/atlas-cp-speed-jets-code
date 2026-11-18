"""Project the exact approved payload's release NNJvt update into a test asset."""
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from exporters.small_jvt import DEFAULT_GROUP_DATA, SOURCE_FILES, WINDOW, encode_cut_map


def main(canonical, output):
    payload = json.loads(canonical.read_bytes())
    update, = (row for row in payload["corrections"] if row["name"] == "small_nnjvt_update")
    config = update["data"]["config"]
    input_order = ["pt", "eta", "sum_pt_trk", "jvf_corr", "vertex_kind", "feature_status"]
    if ([arg["name"] for arg in update["inputs"]] != input_order
        or [arg["type"] for arg in update["inputs"]] != ["real"] * 4 + ["int"] * 2
        or update["output"]["shape"] != [4]
        or update["data"]["nodetype"] != "extension"
        or update["data"]["type"] != "org.atlas.jets.nnjvt_update"
        or update["data"]["version"] != 1
        or update["data"]["output_shape"] != [4]
        or update["data"]["children"]):
        raise ValueError("Unexpected release NNJvt update signature")
    sources = {}
    for role in ("network", "cuts"):
        relative, expected = SOURCE_FILES[role]
        path = DEFAULT_GROUP_DATA / relative
        contents = path.read_bytes()
        digest = hashlib.sha256(contents).hexdigest()
        if digest != expected:
            raise ValueError(f"Release NNJvt {role} digest differs")
        sources[role] = {"path": str(path), "sha256": digest, "bytes": len(contents)}
        if role == "network" and config["network_json"].encode("utf-8") != contents:
            raise ValueError("Embedded network differs from the pinned model")
        if role == "cuts" and config["cut_map"] != encode_cut_map(contents):
            raise ValueError("Embedded cut map differs from the pinned FixedEffPt map")
    if config["max_pt"] != WINDOW["max_pt"] or config["mev_per_gev"] != 1000.0:
        raise ValueError("The fixture expects the frozen run-3 NNJvt update scope")
    def put(name, data):
        with (output / name).open("x") as stream:
            json.dump(data, stream, allow_nan=False, indent=2)
            stream.write("\n")
    put("model-receipt.json", sources)
    put("cuts-and-network.json", config)
    put("nnjvt-update.json", {"schema_version": payload["schema_version"], "corrections": [update]})


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: prepare-payload.py CANONICAL_CORRECTIONSET EXISTING_OUTPUT_DIR")
    main(*(Path(arg).resolve(strict=True) for arg in sys.argv[1:]))
