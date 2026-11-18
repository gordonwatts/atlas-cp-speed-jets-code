"""Build the local wheel and visible PEP 723 demonstration from pinned artifacts."""

import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Annotated
import zipfile

import typer

APP = typer.Typer(add_completion=False)
CHECKOUT = Path(__file__).resolve().parent.parent
NATIVE_TAG = "cp314-cp314-linux_x86_64"
BUNDLED_COMPILER = ("libstdc++.so.6", "libgcc_s.so.1")
ROOT_NAMES = (
    "libROOT", "libCore.so", "libRIO.so", "libHist.so", "libTree.so",
    "libCling.so", "libMathCore.so", "libPhysics.so", "libcppyy", "libCPyCppyy",
    "libxAOD", "libJetCalibTools", "libJetUncertainties", "libAsgTools", "liblwtnn.so",
)
LICENSES = (
    "AnalysisBase-LICENSE.txt", "GCC-COPYING3", "GCC-COPYING.RUNTIME", "lwtnn-2.14.1-LICENCE",
    "ROOT-LICENSE", "ROOT-LGPL2_1.txt", "ROOT-adaptation-NOTICE.md",
)
CLEAN_ELF_ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}


def file_digest(path: Path) -> bytes:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").digest()


def identify(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": file_digest(path).hex()}


def replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise ValueError(f"Expected one visible-script insertion site: {before!r}")
    return source.replace(before, after, 1)


def use_compact_payload(source: str) -> str:
    source = replace_once(source, "from pathlib import Path\n",
                          "from pathlib import Path\nfrom tempfile import TemporaryDirectory\n")
    source = replace_once(
        source, "from atlas_jet_demo.runtime import",
        "from atlas_jet_demo.payload_ranges import materialize\nfrom atlas_jet_demo.runtime import",
    )
    source = replace_once(
        source, "    scope: Scope = Scope.both,\n",
        '    compact_payload: Annotated[bool, typer.Option(\n'
        '        "--compact-payload", help="Load only the selected collection from the packaged payload",\n'
        '    )] = False,\n'
        '    scope: Scope = Scope.both,\n',
    )
    source = replace_once(
        source, '    payload = payload or DATA / "jets.json"',
        '    canonical = DATA / "jets.json"\n'
        '    if compact_payload and payload is not None and payload.resolve() != canonical.resolve():\n'
        '        raise typer.BadParameter("--compact-payload requires the bundled CorrectionSet")\n'
        '    payload = payload or canonical',
    )
    return replace_once(
        source, "        nodes, buffers = load_corrections(provider, payload)",
        '        if compact_payload and scope.value in ("small", "large"):\n'
        '            # Keep the canonical bundle once. Load a temporary file with the matching nodes.\n'
        '            with TemporaryDirectory(prefix="atlas-jet-payload-") as scratch:\n'
        '                projected = Path(scratch) / "jets.json"\n'
        '                written = materialize(payload, DATA / "payload-index.json", scope.value, projected)\n'
        '                typer.echo("Payload materialized SHA-256: " + written["sha256"])\n'
        '                typer.echo("Payload materialized bytes: " + str(written["bytes"]))\n'
        '                nodes, buffers = load_corrections(provider, projected)\n'
        '        else:\n'
        '            nodes, buffers = load_corrections(provider, payload)',
    )


def build_visible_script(wheel: Path, version: str, output: Path, compact: bool = False) -> None:
    source = (CHECKOUT / "columnar/jets.py").read_text()
    source = replace_once(source, "# dependencies = [", f'# dependencies = [\n#   "correctionlib=={version}",')
    source = replace_once(
        source, "# ///\n\"\"\"Read original PHYSLITE",
        '#\n# [tool.uv.sources]\n# correctionlib = { path = "' + wheel.name
        + '" }\n# ///\n"""Read original PHYSLITE'
    )
    source = replace_once(
        source,
        "Use this campaign's CorrectionLib wheel (uv run --with /path/to/wheel.whl) and\n"
        "native provider. Pin the process to one CPU with taskset. Stored controls need\n"
        "neither a provider nor a payload.",
        "Run with uv run jets_demo.py INPUT OUTPUT and keep the local wheel beside this script.\n"
        "The default evaluates all configured small- and large-R states. "
        "The ROOT-free implementation preserves the deterministic positive-seed TRandom3 stream for JER.",
    )
    source = replace_once(source, "import awkward as ak",
                          "from atlas_jet_demo.runtime import DATA, PROVIDER, audit_runtime, prepare_runtime\n\n"
                          "prepare_runtime()\n\nimport awkward as ak")
    source = replace_once(
        source,
        "CHECKOUT = Path(__file__).resolve().parents[1]\n"
        "if not __package__ and str(CHECKOUT) not in sys.path:\n"
        "    sys.path.insert(0, str(CHECKOUT))\n\n"
        "from columnar.inputs import (", "from atlas_jet_demo.inputs import (",
    )
    source = replace_once(source, "import sys\n", "")
    source = replace_once(source, "from jet_transport import JetBuffers",
                          "from atlas_jet_demo.transport import JetBuffers")
    source = replace_once(source, 'CHECKOUT / "config/jet-states.json"', 'DATA / "jet-states.json"')
    source = replace_once(source, 'CHECKOUT / "config/jet-histogram.json"', 'DATA / "jet-histogram.json"')
    source = replace_once(source, "mode: Mode = Mode.nominal", "mode: Mode = Mode.full")
    source = replace_once(
        source, '    """Histogram jets directly from the original PHYSLITE CollectionTree."""\n'
        "    check_execution()",
        '    """Histogram jets directly from the original PHYSLITE CollectionTree."""\n'
        "    # Respect an existing taskset assignment. A direct launch chooses one allowed CPU.\n"
        "    allowed = os.sched_getaffinity(0)\n"
        "    if len(allowed) > 1:\n"
        "        os.sched_setaffinity(0, {min(allowed)})\n"
        "    check_execution()\n"
        "    provider = provider or PROVIDER\n"
        '    payload = payload or DATA / "jets.json"'
    )
    source = replace_once(
        source, '    with output.open("xb") as stream:\n        np.savez(stream, **results)',
        '    if mode != Mode.stored:\n        audit_runtime()\n    with output.open("xb") as stream:\n'
        "        np.savez(stream, **results)"
    )
    if compact:
        source = use_compact_payload(source)
    compile(source, output.as_posix(), "exec")
    output.write_text(source)


def unpack_wheel(wheel: Path, destination: Path) -> tuple[Path, str]:
    with zipfile.ZipFile(wheel) as archive:
        for item in archive.infolist():
            name = Path(item.filename)
            if name.is_absolute() or ".." in name.parts or stat_type(item.external_attr) == 0o120000:
                raise ValueError(f"Unexpected base wheel member: {item.filename}")
        archive.extractall(destination)
    info, = destination.glob("correctionlib-*.dist-info")
    metadata = (info / "METADATA").read_text()
    match = re.search(r"(?m)^Version: (\S+)$", metadata)
    if not match or f"Tag: {NATIVE_TAG}" not in (info / "WHEEL").read_text().splitlines():
        raise ValueError("Expected the pinned CPython 3.14 Linux x86-64 evaluator wheel")
    return info, match.group(1)


def stat_type(external_attr: int) -> int:
    return (external_attr >> 16) & 0o170000


def stock_licenses(info: Path, license_dir: Path) -> list[str]:
    subtree = info / "licenses" / "atlas-jets"
    subtree.mkdir(parents=True)
    members = []
    for name in LICENSES:
        shutil.copyfile(license_dir / name, subtree / name)
        members.append("atlas-jets/" + name)
    core = CHECKOUT / "source/correctionlib"
    patterns = {
        "cpp-peglib": ("LICENSE",),
        "eigen": ("COPYING.*",),
        "lwtnn": ("LICENSE",),
        "pcg-cpp": ("LICENSE*",),
        "pybind11": ("LICENSE",),
        "rapidjson": ("license.txt",),
        "xxhash": ("LICENSE",),
    }
    for module, globs in patterns.items():
        target = subtree / f"core-{module}"
        target.mkdir()
        for pattern in globs:
            for path in sorted((core / module).glob(pattern)):
                if path.is_file():
                    shutil.copyfile(path, target / path.name)
                    members.append(f"atlas-jets/core-{module}/{path.name}")
    return members


def elf_command(tool: str, *args: str) -> str:
    return subprocess.check_output([tool, *args], env=CLEAN_ELF_ENV, text=True, stderr=subprocess.STDOUT)


def relocate_and_audit(patchelf: Path, stage: Path, audit_dir: Path) -> dict:
    native = stage / "correctionlib"
    extension, = native.glob("_core.cpython-314-*.so")
    library_dir = native / "lib"
    targets = [extension, *sorted(path for path in library_dir.glob("*.so*") if path.is_file())]
    audit = {}
    audit_dir.mkdir()
    for binary in targets:
        runpath = "$ORIGIN/lib" if binary == extension else "$ORIGIN"
        subprocess.run([str(patchelf), "--set-rpath", runpath, str(binary)],
                       check=True, env=CLEAN_ELF_ENV)
    forbidden = set()
    for binary in targets:
        readelf = elf_command("readelf", "--dynamic", str(binary))
        versions = elf_command("readelf", "--version-info", str(binary))
        dependency = elf_command("ldd", str(binary))
        if "not found" in dependency or "/cvmfs" in dependency or any(name in dependency for name in ROOT_NAMES):
            forbidden.add(binary.name)
        if "/cvmfs" in readelf:
            raise ValueError(f"Absolute release linkage survived relocation in {binary}")
        if binary not in (library_dir / "libgcc_s.so.1", library_dir / "libstdc++.so.6"):
            for name in BUNDLED_COMPILER:
                if name in dependency and f"{library_dir}/{name}" not in dependency:
                    raise ValueError(f"{binary} did not resolve {name} to the wheel")
        for suffix, content in (("dynamic", readelf), ("versions", versions), ("ldd", dependency)):
            (audit_dir / f"{binary.name}.{suffix}.txt").write_text(content)
        needs = versions.split("Version needs section")[-1] if "Version needs section" in versions else ""
        glibc = sorted(set(re.findall(r"Name: GLIBC_(\d+(?:\.\d+)+)", needs)),
                       key=lambda value: tuple(int(x) for x in value.split(".")))
        audit[str(binary.relative_to(stage))] = {**identify(binary), "glibc_required_versions": glibc}
    if forbidden:
        raise ValueError(f"Unexpected native dependency in {sorted(forbidden)}; see {audit_dir}")
    return audit


def pack(stage: Path, info: Path, path: Path) -> None:
    records = io.StringIO()
    writer = csv.writer(records, lineterminator="\n")
    record_path = info / "RECORD"
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for item in sorted(stage.rglob("*")):
            if not item.is_file() or item == record_path:
                continue
            relative = item.relative_to(stage).as_posix()
            kind = zipfile.ZipInfo(relative)
            kind.compress_type = zipfile.ZIP_DEFLATED
            kind.external_attr = (0o100755 if item.relative_to(stage).is_relative_to("correctionlib/lib")
                                  else 0o100644) << 16
            with item.open("rb") as raw, target.open(kind, "w", force_zip64=item.stat().st_size > 1 << 30) as member:
                shutil.copyfileobj(raw, member)
            digest = base64.urlsafe_b64encode(file_digest(item)).decode().rstrip("=")
            writer.writerow((relative, f"sha256={digest}", item.stat().st_size))
        name = record_path.relative_to(stage).as_posix()
        writer.writerow((name, "", ""))
        target.writestr(name, records.getvalue())


@APP.command()
def build(
    base_wheel: Path,
    provider: Path,
    payload: Path,
    output: Path,
    compiler_dir: Path,
    license_dir: Path,
    patchelf: Path,
    variant: str = "atlasjets1",
    scope_index: Annotated[Path | None, typer.Option(help="Include the index for opt-in compact loading")] = None,
) -> None:
    """Assemble a local patched-precision CorrectionLib wheel and a visible runnable script."""
    if re.fullmatch(r"atlasjets\d+", variant) is None:
        raise ValueError("Expected an atlasjetsN suffix")
    output = output.resolve()
    provider = provider.resolve(strict=True)
    payload = payload.resolve(strict=True)
    payload_identity = identify(payload)
    if scope_index is not None:
        layout = json.loads(scope_index.read_text())
        if layout.get("format") != 1 or layout["canonical"] != payload_identity:
            raise ValueError("Scope index does not describe the bundled canonical CorrectionSet")
    output.mkdir()
    with tempfile.TemporaryDirectory(prefix="stage-", dir=output) as temporary:
        stage = Path(temporary)
        info, source_version = unpack_wheel(base_wheel, stage)
        version = source_version + "." + variant
        relocated = stage / f"correctionlib-{version}.dist-info"
        info.rename(relocated)
        metadata = (relocated / "METADATA").read_text()
        metadata = replace_once(metadata, f"Version: {source_version}\n", f"Version: {version}\n")
        metadata = re.sub(r"(?m)^Requires-Python: .+$", "Requires-Python: >=3.14,<3.15", metadata)
        header, separator, body = metadata.partition("\n\n")
        files = stock_licenses(relocated, license_dir)
        header += "".join(f"\nLicense-File: {name}" for name in files)
        (relocated / "METADATA").write_text(header + separator + body)
        # Keep the import-reported version consistent with the installed wheel.
        version_file = stage / "correctionlib/version.py"
        version_source = version_file.read_text().replace(source_version, version).replace(
            repr(source_version.split("+", 1)[1]), repr(version.split("+", 1)[1])
        )
        version_file.write_text(version_source)

        library_dir = stage / "correctionlib/lib"
        shutil.copyfile(provider, library_dir / "libjets.so")
        for name in BUNDLED_COMPILER:
            shutil.copyfile(compiler_dir / name, library_dir / name)
        assets = stage / "atlas_jet_demo"
        assets.mkdir()
        (assets / "__init__.py").write_text('"""Packaged ATLAS jet correction inputs and runtime."""\n')
        for origin, dest in (
            (CHECKOUT / "packaging/runtime.py", assets / "runtime.py"),
            (CHECKOUT / "columnar/inputs.py", assets / "inputs.py"),
            (CHECKOUT / "jet_transport.py", assets / "transport.py"),
        ):
            shutil.copyfile(origin, dest)
        data = assets / "data"
        data.mkdir()
        provenance = payload.with_name("provenance.json")
        for origin, dest in (
            (payload, data / "jets.json"),
            (provenance, data / "provenance.json"),
            (CHECKOUT / "config/jet-states.json", data / "jet-states.json"),
            (CHECKOUT / "config/jet-histogram.json", data / "jet-histogram.json"),
        ):
            shutil.copyfile(origin, dest)
        if scope_index is not None:
            shutil.copyfile(scope_index, data / "payload-index.json")
            shutil.copyfile(CHECKOUT / "exporters/payload_ranges.py", assets / "payload_ranges.py")
        build_identity = {
            "tag": NATIVE_TAG,
            "original_wheel": {"name": base_wheel.name, **identify(base_wheel)},
            "provider": {"name": provider.name, **identify(provider)},
            "payload": payload_identity,
            "payload_provenance": identify(provenance),
            "compiler_inputs": {name: identify(compiler_dir / name) for name in BUNDLED_COMPILER},
            "columnar_source": identify(CHECKOUT / "columnar/jets.py"),
            "input_transport": identify(CHECKOUT / "jet_transport.py"),
            "native_members": relocate_and_audit(patchelf.resolve(strict=True), stage, output / "elf-audit"),
            "git_commit": subprocess.check_output(["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"],
                                                 text=True).strip(),
        }
        if scope_index is not None:
            build_identity["scope_index"] = identify(scope_index)
            build_identity["scope_projector"] = identify(assets / "payload_ranges.py")
        (data / "build.json").write_text(json.dumps(build_identity, indent=2) + "\n")
        wheel = output / f"correctionlib-{version}-{NATIVE_TAG}.whl"
        script = output / "jets_demo.py"
        build_visible_script(wheel, version, script, compact=scope_index is not None)
        pack(stage, relocated, wheel)

    (output / "bundle.json").write_text(json.dumps({
        **build_identity, "wheel": {"name": wheel.name, **identify(wheel)},
        "visible_script": {"name": script.name, **identify(script)},
    }, indent=2) + "\n")
    (output / "SHA256SUMS").write_text("".join(
        f"{file_digest(path).hex()}  {path.name}\n" for path in (wheel, script, output / "bundle.json")
    ))
    print(wheel)
    print(script)


if __name__ == "__main__":
    APP()
