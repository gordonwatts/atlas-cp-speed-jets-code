"""Seal per-scope recombination plans for the existing full-file columnar harness."""
from hashlib import sha256
from pathlib import Path
import re
import sys
from typing import Annotated
from zipfile import ZipFile

import typer

from joint_trials import BUILD, COMPILER_LIB, EXPORT, REFERENCE, ROOT, SCOPES, checksums, emit, json_file, sha, verify_payload
from profile_final_demo import stock_endpoint

sys.path.insert(0, str(ROOT / "tools/reference"))
from workflow import open_root


PRECISION = BUILD / "core-precision-002"
PACKED = EXPORT / "eta-jes-packed-001"
PARTITION = EXPORT / "eta-jes-partition-001"
SOURCE_FILES = (
    "columnar/jets.py", "columnar/inputs.py", "columnar/distributions.py", "jet_transport.py",
    "config/jet-states.json", "config/jet-histogram.json", "config/jets-input.json",
    "tools/compare_columnar.py", "tools/screen_trials.sh", "tools/time_columnar_trials.py",
)
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def identity(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def names(values, description):
    if (not isinstance(values, list) or not values
            or any(not isinstance(item, str) or not LABEL.fullmatch(item) for item in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"Expected a nonempty list of distinct {description}")
    return values


def active_scopes(values):
    selected = set(names(values, "scope labels"))
    if selected - set(SCOPES):
        raise ValueError(f"Unknown scope: {sorted(selected - set(SCOPES))}")
    return selected


def precision_runtime():
    python = PRECISION / "venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError(python)
    site = PRECISION / "venv/lib/python3.14/site-packages"
    sdk = site / "correctionlib"
    core = sdk / "lib/libcorrectionlib.so"
    wheel_receipt = PRECISION / "wheel.sha256"
    (wheel, expected), = checksums(wheel_receipt).items()
    verified_wheel = identity(wheel)
    if verified_wheel["sha256"] != expected:
        raise ValueError("Precision wheel disagrees with its archived receipt")
    with ZipFile(wheel) as archive:
        bindings = [name for name in archive.namelist() if re.fullmatch(
            r"correctionlib/_core\.cpython-314-[^/]+\.so", name)]
        if len(bindings) != 1:
            raise ValueError("Expected exactly one CPython extension in the precision wheel")
        binding = site / bindings[0]
        actual = {path: identity(path) for path in (core, binding)}
        for path, record in actual.items():
            if record["sha256"] != sha256(archive.read(path.relative_to(site).as_posix())).hexdigest():
                raise ValueError(f"The installed evaluator differs from the precision wheel: {path}")
    compiled = {path: identity(path) for path in (COMPILER_LIB / "libstdc++.so.6", COMPILER_LIB / "libgcc_s.so.1")}
    record = {
        "python": identity(python), "python_real_path": str(python.resolve(strict=True)), "sdk": str(sdk),
        "wheel": verified_wheel, "wheel_receipt": identity(wheel_receipt), "installed_native": list(actual.values()),
        "compiler_native": list(compiled.values()),
    }
    return {"python": str(python), "sdk": str(sdk)}, record, core, actual[core]["sha256"]


def provider_arm(label, spec, core, core_sha):
    unexpected = set(spec) - {"build", "flags", "scopes", "signed"}
    if unexpected:
        raise ValueError(f"Unknown variant fields for {label}: {sorted(unexpected)}")
    build = (ROOT / spec["build"]).resolve(strict=True)
    if not build.is_relative_to(BUILD):
        raise ValueError(f"Provider build must come from the authoritative build results: {build}")
    scopes = active_scopes(spec["scopes"])
    flags = set(names(spec["flags"], "build flags")) if spec["flags"] != [] else set()
    signed = spec.get("signed", False)
    if type(signed) is not bool:
        raise ValueError(f"{label}.signed must be boolean")
    names_on_disk = (build / "build-choices.txt").read_text().splitlines()
    if names_on_disk == ["baseline"]:
        names_on_disk = []
    if (build / "exit-code").read_text().strip() != "0" or sorted(flags) != sorted(names_on_disk):
        raise ValueError(f"Declared provider flags differ from the completed build: {build}")
    macros = set((build / "build-defines.txt").read_text().splitlines())
    if ("-DJETS_PORTABLE" not in macros
            or ("-DJETS_UNSIGNED_SIGN_PAIRS" in macros) != ("unsigned-sign-pairs" in flags)
            or (signed and "-DJETS_UNSIGNED_SIGN_PAIRS" not in macros)):
        raise ValueError(f"The provider macros do not support the requested portable signed-state mode: {label}")
    if "rng-cache-diagnostics" in flags:
        raise ValueError(f"Timing requires a provider without the cache-counter fixture: {label}")

    provider = build / "libjets.so"
    native_receipt = build / "native.sha256"
    sources_receipt = build / "source-inputs.sha256"
    linked = checksums(native_receipt)
    lwtnn = [path for path in linked if path.name == "liblwtnn-stat.a"]
    if len(lwtnn) != 1 or set(linked) != {provider, core, lwtnn[0]} or linked[core] != core_sha:
        raise ValueError(f"The archived provider does not identify the precision SDK and lwtnn archive: {label}")
    binary = identity(provider)
    static = identity(lwtnn[0])
    if binary["sha256"] != linked[provider] or static["sha256"] != linked[lwtnn[0]]:
        raise ValueError(f"Archived native bytes changed: {label}")
    revision = (build / "source-revision.txt").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"Missing provider source revision: {label}")
    source_map = checksums(sources_receipt)
    if not source_map:
        raise ValueError(f"Empty provider source input receipt: {label}")
    options = ["--reuse-nominal-nnjvt-update"]
    if signed:
        options.append("--reuse-uncertainty-sign-pairs")
    return {
        "provider": str(provider), "scopes": scopes, "options": options,
        "build": {
            "provider": binary, "static_lwtnn": static, "core_sha256": core_sha, "flags": sorted(flags),
            "signed": signed, "source_revision": revision,
            "source_inputs": {str(path): digest for path, digest in source_map.items()},
            "receipts": {name: identity(build / name) for name in (
                "native.sha256", "source-inputs.sha256", "source-revision.txt", "build-choices.txt",
                "build-defines.txt", "rng-cache-slots.txt", "needed.txt", "libraries.txt",
            )},
        },
    }


def packed_inputs():
    packing = PACKED / "packing.json"
    index = PARTITION / "payload-index.json"
    manifest = PARTITION / "manifest.json"
    canonical = PACKED / "jets.json"
    provenance = PACKED / "provenance.json"
    packed = json_file(packing)
    layout = json_file(index)
    partition = json_file(manifest)
    if layout["format"] != 1 or not partition["raw_node_identity_verified"]:
        raise ValueError("Expected the verified prebuilt packed partition")
    combined = identity(canonical)
    assets = {"both": combined}
    proofs = {"both": identity(provenance)}
    if json_file(provenance)["sha256"] != combined["sha256"] or proofs["both"]["sha256"] != partition["provenance"]["sha256"]:
        raise ValueError("Packed payload and archived provenance differ")
    for other in (packed["packed"], partition["canonical"], layout["canonical"]):
        if any(other[field] != combined[field] for field in ("bytes", "sha256")):
            raise ValueError("The partition or packing receipt differs from the canonical payload")
    for scope in ("small", "large"):
        relative = partition["native_load_paths"][scope]
        path = (PARTITION / relative).resolve(strict=True)
        if not path.is_relative_to(PARTITION) or path.parent != PARTITION / "scopes" / scope:
            raise ValueError(f"Invalid archived partition path for {scope}")
        asset = identity(path)
        assets[scope] = asset
        for expected in (layout["scopes"][scope], partition["artifacts"][relative]):
            if any(expected[field] != asset[field] for field in ("bytes", "sha256")):
                raise ValueError(f"The prebuilt scope payload differs from its checked archive: {scope}")
        proofs[scope] = identity(path.with_name("provenance.json"))
        if proofs[scope]["sha256"] != proofs["both"]["sha256"]:
            raise ValueError(f"The prebuilt scope has different packed provenance: {scope}")
    return assets, proofs, {path.name: identity(path) for path in (packing, index, manifest)}


def independent_source(scopes):
    source = json_file(ROOT / "config/jets-input.json")
    original = Path(source["path"]).resolve(strict=True)
    verify_payload(original, source)
    # The file and tree metadata prove the UUID and interval without reading event arrays.
    with open_root(original) as root_file:
        if (str(root_file.file.uuid) != source["tfile_uuid"]
                or root_file[source["collection_tree"]].num_entries != 20000
                or source["entry_interval"] != [0, 20000]):
            raise ValueError("Original PHYSLITE metadata differs from the frozen full interval")
    observer = REFERENCE / "full-clean-001"
    verification = identity(observer / "verification.json")
    independent = json_file(Path(verification["path"]))
    if independent["events"] != 20000 or independent["source_file_uuid"] != source["tfile_uuid"]:
        raise ValueError("Independent observer does not cover the frozen original file")
    authority = {}
    for scope in scopes:
        prefix = "combined" if scope == "both" else scope
        authority[scope] = stock_endpoint(REFERENCE / f"full-file-pilot-{prefix}-002", scope, original, source, sha)
    return source, original, observer / "reference.root", verification, authority


def read_specification(specification, core, core_sha):
    config = json_file(specification)
    if set(config) != {"variants", "comparisons"} or not config["variants"] or not config["comparisons"]:
        raise ValueError("Specify nonempty variants and comparisons mappings")
    variants = {}
    for label, spec in config["variants"].items():
        if not LABEL.fullmatch(label):
            raise ValueError(f"Invalid variant label: {label}")
        variants[label] = provider_arm(label, spec, core, core_sha)
    comparisons = {}
    for label, entry in config["comparisons"].items():
        if not LABEL.fullmatch(label) or set(entry) != {"control", "candidate", "scopes"}:
            raise ValueError(f"Invalid comparison declaration: {label}")
        left, right = entry["control"], entry["candidate"]
        if left == right or not {left, right} <= variants.keys():
            raise ValueError(f"Comparison must name two different known variants: {label}")
        control, candidate = variants[left], variants[right]
        if (control["build"]["provider"]["sha256"] == candidate["build"]["provider"]["sha256"]
                and control["options"] == candidate["options"]):
            raise ValueError(f"Requested variants implement the same effective configuration: {label}")
        scopes = active_scopes(entry["scopes"])
        if scopes - (control["scopes"] & candidate["scopes"]):
            raise ValueError(f"Comparison declares a scope unavailable from either arm: {label}")
        comparisons[label] = {"control": left, "candidate": right, "scopes": [s for s in SCOPES if s in scopes]}
    return variants, comparisons


def main(specification: Path, output: Path,
         rounds: Annotated[int, typer.Option(min=1, max=4)] = 4,
         screen_events: Annotated[int, typer.Option(min=1, max=20000)] = 32):
    """Write one existing-runner plan per requested scope. Do not launch any analyses."""
    if rounds not in (1, 4):
        raise typer.BadParameter("Use one diagnostic round or four rotated rounds", param_hint="--rounds")
    if output.exists():
        raise FileExistsError(output)
    specification, output = specification.resolve(strict=True), output.resolve()
    runtime, sdk, core, core_sha = precision_runtime()
    variants, comparisons = read_specification(specification, core, core_sha)
    scopes = [s for s in SCOPES if any(s in entry["scopes"] for entry in comparisons.values())]
    assets, proofs, packing = packed_inputs()
    source, original, observer, stock_proof, authority = independent_source(scopes)
    source_hashes = {str(ROOT / name): sha(ROOT / name) for name in SOURCE_FILES}
    output.mkdir()
    plans = {}
    for scope in scopes:
        active = {name: arm for name, arm in variants.items() if scope in arm["scopes"]}
        if len(active) < 2:
            raise ValueError(f"Need at least two applicable variants for {scope}")
        authority_hist = authority[scope]["histogram"]
        plan = {
            "experiment": "final-native-recombination", "rounds": rounds, "scopes": [scope], "events": 20000,
            "input": str(original), "input_identity": source, "observer": str(observer),
            "authority": {scope: authority_hist["path"]},
            "script": str(ROOT / "columnar/jets.py"),
            "script_sources": {**source_hashes, authority_hist["path"]: authority_hist["sha256"]},
            "screen_events": screen_events, "screen_each_scope": True,
            "compiler_only_runtime": True, "require_matching_branches": True,
            "variants": {name: {**runtime, "provider": arm["provider"], "scopes": [scope], "options": arm["options"],
                                "payloads": {scope: assets[scope]["path"]},
                                "provenances": {scope: proofs[scope]["path"]}} for name, arm in active.items()},
        }
        path = output / f"{scope}.json"
        emit(path, plan)
        plans[scope] = identity(path)
    index = {
        "definition": identity(specification), "rounds": rounds, "screen_events": screen_events,
        "plans": plans, "precision_runtime": sdk, "physical_source": source, "verified_original_path": str(original),
        "observer_verification": stock_proof, "stock_authority": authority, "packing_receipts": packing,
        "payloads": assets, "payload_provenance": proofs,
        "variants": {name: {**arm["build"], "scopes": [s for s in SCOPES if s in arm["scopes"]]}
                     for name, arm in variants.items()},
        "comparisons": {label: {**entry, "plans": {scope: plans[scope]["path"] for scope in entry["scopes"]}}
                        for label, entry in comparisons.items()},
        "policy": "Run each scope separately using the existing alternating fresh-process harness. "
                  "Four rounds exclude the first. One round provides a diagnostic without retained timing means.",
    }
    emit(output / "index.json", index)


if __name__ == "__main__":
    typer.run(main)
