"""Common offline ATLAS calibration-area resolution and source provenance."""

import hashlib
from pathlib import Path


def versioned_calibration_name(requested, area):
    prefix = "JetCalibTools/"
    if not requested.startswith(prefix):
        raise ValueError(f"expected a {prefix} configuration path, got {requested!r}")
    return f"{prefix}CalibArea-{area}/{requested[len(prefix):]}"


def resolve(resolver, requested):
    found = str(resolver.FindCalibFile(requested))
    if not found or not Path(found).is_file():
        raise FileNotFoundError(f"PathResolver could not resolve {requested}")
    return Path(found)


def source_record(requested, path):
    content = path.read_bytes()
    return {
        "path_resolver_name": requested,
        "resolved_path": str(path),
        "real_path": str(path.resolve()),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def read_config(environment, path, level):
    if environment.ReadFile(str(path), level) != 0:
        raise ValueError(f"TEnv could not read {path}")
