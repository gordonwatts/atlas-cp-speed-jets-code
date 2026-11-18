"""Archive exact bytes and accepted metadata from one completed jet qualification."""

from dataclasses import dataclass
from errno import EACCES, ENOSYS, EOPNOTSUPP, EPERM, EXDEV
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

import numpy as np
import typer

from profile_final_demo import checked_installation, histogram_comparison, stock_endpoint, validate_run
from profile_native import FORBIDDEN, digest, verify_loaded, wheel_identity


ROOT = Path(__file__).resolve().parents[1]
REPORT = Path("/mnt/c/Users/gwatts/OneDrive/Documents/ATLAS/Computing/AMG/CorrectionLib/jets-skill-c8a2374-20261117")
FINAL = Path("evidence/final")
SCOPES = ("both", "small", "large")
MODES = ("stored", "nominal", "full")
EVENTS = 20_000
SHIPPED = ("pip-canonical-full", "pip-indexed-full", "pip-indexed-nominal", "uv-indexed-full",
           "pip-selected-default-full")


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_path(value, expected):
    return Path(value).resolve() == expected.resolve()


def check_record(path, record, *, check_path=True):
    if check_path:
        require(same_path(record["path"], path), f"Receipt points to a different file: {path}")
    require(("bytes" not in record or record["bytes"] == path.stat().st_size) and record["sha256"] == digest(path),
            f"Artifact differs from its receipt: {path}")


def checksum_records(path):
    return {Path(file.strip().removeprefix("*")).resolve(): checksum
            for line in path.read_text().splitlines() if line.strip()
            for checksum, file in [line.split(maxsplit=1)]}


def archive_member(directory, name):
    require(Path(name).name == name and name not in {"", ".", ".."}, f"Unsafe archive member name: {name}")
    return directory / name


def histograms(actual, expected, scope, events=EVENTS):
    checks = histogram_comparison(expected, actual, scope, events)
    require(all(checks.values()), f"Saved histogram fields differ for {actual}: "
            + ", ".join(key for key, exact in checks.items() if not exact))


def check_nosys(nominal, full, scope):
    prefixes = {"both": ("sr", "lr"), "small": ("sr",), "large": ("lr",)}[scope]
    state_fields = {f"{prefix}_states" for prefix in prefixes}
    with np.load(nominal, allow_pickle=False) as a, np.load(full, allow_pickle=False) as b:
        fields = set(a.files)
        require(fields == set(b.files) and state_fields <= fields and int(a["events"]) == int(b["events"]) == EVENTS,
                f"NOSYS and full histogram inventories differ: {nominal}")
        for field in state_fields:
            observed, reference = a[field], b[field]
            # Fixed-width Unicode storage expands with the longest systematic name.
            require(observed.dtype.kind == reference.dtype.kind == "U" and observed.shape == (1,)
                    and reference.ndim == 1 and observed[0] == "NOSYS"
                    and np.array_equal(observed, reference[:1]) and np.count_nonzero(reference == "NOSYS") == 1,
                    f"Expected exactly one leading NOSYS category in {field} for {nominal}")
        for field in fields - state_fields:
            reference = b[field] if field in ("edges", "events") else b[field][..., :1]
            observed = a[field]
            require(observed.dtype == reference.dtype and observed.shape == reference.shape
                    and observed.tobytes() == reference.tobytes(), f"NOSYS field {field} differs in {nominal}")


def check_map(path, shipped_native, installation):
    record = read(path)
    libraries = {Path(item).resolve() for item in record["libraries"]}
    require(not any(item.is_relative_to("/cvmfs") or item.name.startswith(FORBIDDEN) for item in libraries),
            f"Stock or ROOT library appeared in a visible run: {path}")
    for name, expected in shipped_native.items():
        matches = {item for item in libraries if item.name == name
                   and (not name.startswith("_core.") or item.parent.name == "correctionlib")}
        require(len(matches) == 1, f"Could not identify {name} in {path}")
        actual = matches.pop()
        require(actual.is_relative_to(installation) and digest(actual) == expected["sha256"],
                f"Visible run loaded a different {name}: {actual}")


@dataclass(frozen=True)
class File:
    source: Path
    destination: Path
    size: int
    sha256: str


class Archive:
    def __init__(self, report):
        self.report = report
        self.files = {}

    def add(self, source, destination):
        source = source.resolve(strict=True)
        require(source.is_file(), f"Missing regular evidence file: {source}")
        require(destination.is_relative_to(FINAL) or destination.is_relative_to("delivery"),
                f"Invalid report-relative path: {destination}")
        entry = File(source, destination, source.stat().st_size, digest(source))
        previous = self.files.setdefault(destination, entry)
        require(entry == previous, f"Conflicting archive sources: {destination}")
        return entry

    def members(self, directory, destination, names, *, optional=()):
        for name in names:
            self.add(directory / name, destination / name)
        for name in optional:
            path = directory / name
            if path.is_file():
                self.add(path, destination / name)

    def target(self, relative):
        require(not relative.is_absolute() and ".." not in relative.parts, f"Unsafe report path: {relative}")
        target = self.report
        for part in relative.parts:
            target /= part
            require(not target.is_symlink(), f"Refusing to follow an archive destination symlink: {target}")
        return target

    def publish(self, specification, output, wheel):
        entries = sorted(self.files.values(), key=lambda item: item.destination.as_posix())
        manifest = (json.dumps({
            "schema": 1, "qualification_specification": str(specification),
            "qualification_output": str(output), "wheel_sha256": digest(wheel),
            "artifacts": [{"source": str(item.source), "destination": item.destination.as_posix(),
                           "bytes": item.size, "sha256": item.sha256} for item in entries],
        }, indent=2, allow_nan=False) + "\n").encode()
        manifest_target = self.target(FINAL / "manifest.json")
        manifest_digest = sha256(manifest).hexdigest()
        # Validate existing destinations, including the completion manifest, before adding files.
        for entry in entries:
            existing(self.target(entry.destination), entry.size, entry.sha256)
        existing(manifest_target, len(manifest), manifest_digest)
        for entry in entries:
            with entry.source.open("rb") as stream:
                publish(self.target(entry.destination), stream, entry.size, entry.sha256)
        publish(manifest_target, BytesIO(manifest), len(manifest), manifest_digest)
        return len(entries), manifest_target


def existing(target, size, checksum):
    if target.is_symlink():
        raise FileExistsError(f"Archive destination is a symlink: {target}")
    if not target.exists():
        return False
    if not (target.is_file() and target.stat().st_size == size and digest(target) == checksum):
        raise FileExistsError(f"Archive destination already contains different bytes: {target}")
    return True


def publish(target, source, size, checksum):
    if existing(target, size, checksum):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            copyfileobj(source, stream)
            stream.flush()
            os.fsync(stream.fileno())
        require(temporary.stat().st_size == size and digest(temporary) == checksum, f"Source changed before copying {target}")
        try:
            os.link(temporary, target)
        except FileExistsError:
            existing(target, size, checksum)
        except OSError as error:
            if error.errno not in {EACCES, ENOSYS, EOPNOTSUPP, EPERM, EXDEV}:
                raise
            # Some mounted report directories do not support hard links.
            # Keep the same non-overwrite behavior with an exclusive file.
            try:
                stream = target.open("xb")
            except FileExistsError:
                existing(target, size, checksum)
            else:
                try:
                    with stream, temporary.open("rb") as staged:
                        copyfileobj(staged, stream)
                        stream.flush()
                        os.fsync(stream.fileno())
                    require(existing(target, size, checksum), f"Missing copied file {target}")
                except BaseException:
                    target.unlink()
                    raise
    finally:
        if temporary is not None:
            temporary.unlink()


def collect_direct(archive, output, packaged, defaults, partition, observer, original, uuid):
    packaged_provider = Path(packaged["native"]["libjets.so"]["path"])
    packaged_core = Path(packaged["native"]["libcorrectionlib.so"]["path"])
    packaged_payload = Path(packaged["data"]["jets.json"]["path"])
    compact = set(defaults["compact_payload_default_scopes"])
    signed = set(defaults.get("reuse_uncertainty_sign_pair_scopes", []))
    index = read(Path(packaged["workflow"]["data/payload-index.json"]["path"]))["scopes"]
    for scope in SCOPES:
        directory = output / f"C-{scope}"
        path = directory / "comparison.json"
        value = read(path)
        payload = partition / "scopes" / scope / "jets.json" if scope in compact else packaged_payload
        if scope in compact:
            check_record(payload, index[scope], check_path=False)
        require(value["passed"] and (value["scope"], value["mode"], value["stage_limit"]) == (scope, "full", "final")
                and value["entries_compared"] == value["entries_requested"] == EVENTS
                and value["source_file_uuid"] == uuid and same_path(value["source_phySLITE"], original)
                and same_path(value["authority"], observer) and same_path(value["provider"], packaged_provider)
                and same_path(value["runtime"]["provider"], packaged_provider)
                and same_path(value["runtime"]["core"], packaged_core) and same_path(value["payload"], payload)
                and value["reuse_nominal_nnjvt_update"] is defaults["reuse_nominal_nnjvt_update"]
                and value.get("reuse_uncertainty_sign_pairs", False) is (scope in signed)
                and value["execution"] == {"threads": 1, "cpu": 3},
                f"Direct acceptance does not cover the selected {scope} shipment: {path}")
        archive.members(directory, FINAL / directory.name, ("comparison.json", "inputs.sha256", "command.txt", "exit-code"),
                        optional=("revision", "diagnostic-environment.txt", "compare.log", "comparison.time"))
        require((directory / "exit-code").read_text().strip() == "0", f"Direct workflow failed: {directory}")
        hashes = checksum_records(directory / "inputs.sha256")
        for selected in (packaged_provider.resolve(), payload.resolve()):
            require(hashes.get(selected) == digest(selected), f"Direct acceptance input changed: {selected}")
        if scope == "both":
            collect_distributions(archive, directory, value)


def collect_distributions(archive, directory, direct):
    acceptance = direct["endpoint_distributions"]
    distributions = directory / "distributions"
    require(acceptance["passed"] and same_path(acceptance["path"], distributions),
            "Missing accepted combined p4 and energy spectra")
    aliases = read(directory / "observer-aliases.json")
    require(aliases["passed"] and aliases["entries"] == EVENTS and aliases["complete_branches"] == 7189
            and same_path(aliases["authority"], Path(direct["authority"]))
            and str((directory / "comparison.json").resolve()) in aliases["base_comparisons"],
            "Alias or persisted energy proof does not cover the combined observer")
    archive.add(directory / "observer-aliases.json", FINAL / "C-both/observer-aliases.json")
    metadata = read(distributions / "metadata.json")
    require(metadata["entries"] == EVENTS and same_path(metadata["comparison"], directory / "comparison.json")
            and set(metadata["collections"]) == {"sr", "lr"}
            and not metadata["stock_availability"]["large_r_energy"]["persisted"],
            "Distribution metadata does not cover the combined stock observables")
    energy = metadata["energy_method"]
    require(energy["sha256"] == aliases["energy_library_sha256"] and same_path(energy["path"], Path(aliases["energy_library"])),
            "Energy spectra and persisted energy aliases used different helpers")
    check_record(Path(energy["path"]), energy)
    with np.load(distributions / "distributions.npz", allow_pickle=False) as spectra:
        for prefix, collection in metadata["collections"].items():
            states = [row["name"] for row in collection["states"]]
            require(spectra[f"{prefix}_states"].tolist() == states == direct["states"][prefix],
                    f"Distribution state order differs for {prefix}")
            names = ("pt", "eta", "phi", "mass", "energy") if prefix == "sr" else ("pt", "eta", "phi", "mass")
            for name in names:
                observable = collection["observables"][name]
                label = f"{prefix}_{name}"
                counts = spectra[f"{label}_stock_counts"]
                candidate = spectra[f"{label}_direct_counts"]
                states_for_field = states
                if name == "energy":
                    states_for_field = [state for state in states if f"sr_e_{state}" in aliases["mapping"]]
                    require(spectra[observable["state_axis"]].tolist() == states_for_field, "Energy state order differs")
                rows = len(observable["stock_and_direct_bin_mismatches_per_state"])
                require(rows == len(states_for_field) > 0 and not any(observable["stock_and_direct_bin_mismatches_per_state"])
                        and spectra[f"{label}_edges"].tobytes() == np.asarray(observable["edges"], dtype=np.float64).tobytes()
                        and np.array_equal(spectra[f"{label}_stock_nan"], spectra[f"{label}_direct_nan"])
                        and counts.shape == candidate.shape == (rows, len(observable["edges"]) + 1)
                        and counts.dtype == candidate.dtype == np.dtype("int64")
                        and counts.tobytes() == candidate.tobytes()
                        and not spectra[f"{label}_unpaired_chunks"].any()
                        and not spectra[f"{label}_exact_mismatches"].any(),
                        f"Saved exact distribution validation failed: {label}")
    archive.members(distributions, FINAL, ("distributions.npz", "metadata.json"))


def collect_framework(archive, output, paths, defaults, observer, original, uuid, stocks):
    combined = output / "B-combined"
    comparison = read(combined / "observer-comparison.json")
    coverage = read(combined / "reference-coverage.json")
    process = read(combined / "process-verification.json")
    require(comparison["events"] == EVENTS and comparison["branches"] == 7189
            and comparison["service_states"] == 331 and comparison["checked_values"] > 0
            and comparison["exact_observer_parity"] and comparison["exact_histogram_parity"]
            and same_path(comparison["reference"], observer) and coverage["source_file_uuid"] == uuid
            and coverage["events"] == EVENTS and process["passed"] and process["states"] == [[1, "3"]],
            "Combined framework receipt lacks full exact observer, histogram or runtime acceptance")
    combined_files = ("observer-comparison.json", "reference-coverage.json", "process-verification.json", "histograms.npz")
    hashes = checksum_records(combined / "acceptance.sha256")
    for name in combined_files:
        path = combined / name
        require(hashes.get(path.resolve()) == digest(path), f"Framework acceptance file changed: {path}")
    require(hashes.get(stocks["both"].resolve()) == digest(stocks["both"]), "Combined stock histogram changed")
    histograms(combined / "histograms.npz", stocks["both"], "both")
    runtime = checksum_records(combined / "runtime-inputs.sha256")
    core = paths["sdk"] / "lib/libcorrectionlib.so"
    required = (paths["provider"], paths["payload"], core, *sorted(paths["overlay"].glob("*.so")))
    for path in required:
        require(runtime.get(path.resolve()) == digest(path), f"Combined framework did not identify {path}")
    archive.members(combined, FINAL / "B-combined", (*combined_files, "acceptance.sha256", "runtime-inputs.sha256"),
                    optional=("framework-trace.txt", "process-library-maps.txt", "command.txt",
                              "initialized-tools.tsv", "exit-code", "job-exit-code", "input.stat"))

    for scope in ("small", "large"):
        pilot = output / f"B-{scope}"
        recovered = output / f"recovered-{scope}"
        recovery = read(recovered / "recovery.json")
        source_path = pilot / "provenance.json"
        check_record(source_path, recovery["pilot_provenance"])
        source = read(source_path)
        require(recovery["scope"] == scope and recovery["events"] == EVENTS and recovery["endpoint_exact"]
                and not recovery["process_audit_reclassified"] and same_path(recovery["input"], original)
                and source["path"] == "B" and source["pilot"] and source["scope_collections"] == scope
                and source["events"] == EVENTS and same_path(source["input"], original)
                and same_path(source["stock"], stocks[scope].parents[1]),
                f"Scoped framework recovery differs for {scope}")
        payload = (paths["partition"] / "scopes" / scope / "jets.json"
                   if scope in defaults["compact_payload_default_scopes"] else paths["payload"])
        scoped_artifacts = (paths["provider"], payload, core)
        for path in scoped_artifacts:
            require(source["artifact_sha256"].get(str(path.resolve())) == digest(path),
                    f"Scoped framework did not identify {path}")
        all_branches = recovered / "full-ntuple.json"
        check_record(all_branches, recovery["full_ntuple_comparison"])
        branches = read(all_branches)
        require(branches["exact"] and branches["events"] == EVENTS and branches["branches"] > 0 and branches["checked_values"] > 0,
                f"Scoped framework persisted branch receipt failed: {all_branches}")
        for mode in MODES:
            mode_dir = pilot / f"01-{mode}"
            saved = recovered / f"{mode}.npz"
            stock = stocks[scope].parents[1] / f"01-{mode}/histograms.npz"
            receipts = recovery["modes"][mode]
            check_record(mode_dir / "framework.json", receipts["framework_receipt"])
            check_record(saved, receipts["endpoint"])
            check_record(stock, receipts["stock_endpoint"])
            require(read(mode_dir / "framework.json")["exit"] == 0, f"Scoped job did not finish: {mode_dir}")
            histograms(saved, stock, scope)
            archive.members(mode_dir, FINAL / pilot.name / mode_dir.name, ("framework.json",),
                            optional=("framework.log", "histogram.json"))
        check_nosys(recovered / "nominal.npz", recovered / "full.npz", scope)
        archive.members(pilot, FINAL / pilot.name, ("provenance.json",), optional=("summary.json",))
        archive.members(recovered, FINAL / recovered.name, ("recovery.json", "full-ntuple.json", *(f"{mode}.npz" for mode in MODES)))
        if (output / f"B-{scope}-pilot.exit").is_file():
            archive.add(output / f"B-{scope}-pilot.exit", FINAL / pilot.name / "pilot.exit")


def collect_shipped(archive, output, installed, packaged, defaults, original, stocks, demo_sha):
    index = read(Path(packaged["workflow"]["data/payload-index.json"]["path"]))["scopes"]
    for scope in SCOPES:
        directory = output / f"shipped-{scope}"
        path = directory / "comparison.json"
        report = read(path)
        require(report["scope"] == scope and report["events"] == EVENTS and same_path(report["original"], original)
                and report["all_histogram_fields_exact"] and report["matched_nominal_nosys_exact"]
                and report["installation"]["wheel_sha256"] == packaged["sha256"]
                and report["installation"]["script_sha256"] == demo_sha
                and set(report["branch_digests"]) == set(SHIPPED)
                and len(set(report["branch_digests"].values())) == 1,
                f"Shipped pip/uv acceptance is missing scope, branches or identity: {path}")
        check_record(stocks[scope], report["stock_full_histogram"])
        for case in SHIPPED:
            projection = scope != "both" and (case in ("pip-indexed-full", "pip-indexed-nominal", "uv-indexed-full")
                          or case == "pip-selected-default-full" and scope in defaults["compact_payload_default_scopes"])
            wanted = {key: index[scope][key] for key in ("bytes", "sha256")} if projection else {}
            require(report["projection_identities"][case] == wanted, f"Unexpected projected bytes: {scope}/{case}")
            require((directory / f"{case}.exit").read_text().strip() == "0", f"Shipped command failed: {scope}/{case}")
            if case == "pip-indexed-nominal":
                check_nosys(directory / f"{case}.npz", stocks[scope], scope)
            else:
                histograms(directory / f"{case}.npz", stocks[scope], scope)
            check_map(directory / f"{case}-library-maps.json", packaged["native"], installed)
            archive.members(directory, FINAL / directory.name,
                            tuple(f"{case}{suffix}" for suffix in ("-library-maps.json", ".argv.json", ".exit", ".log", ".npz")))
        archive.add(path, FINAL / directory.name / path.name)


def collect_profile(archive, output, plan, packaged, defaults, original, demo, stocks):
    native = output / "native-profile"
    decoded = output / "native-profile-decoded"
    recorded = read(native / "recording.json")
    summary = read(decoded / "summary.json")
    repetitions = plan.get("profile_repeats", 10)
    require(recorded["passed"] and recorded["scope"] == "both" and recorded["events_per_pass"] == EVENTS
            and recorded["repeat"] == repetitions and recorded["event_visits"] == repetitions * EVENTS
            and recorded["measurement"]["worker_cpu_floor_s"] >= 60.0 and recorded["shipped_defaults"] == defaults
            and recorded["calibration_threads"] == 1 and recorded["calibration_cpu"] == 3
            and same_path(recorded["original"]["path"], original) and summary["passed"]
            and summary["artifact"] == recorded["wheel"] and summary["samples"]["processing"] > 0
            and same_path(summary["recording"], native / "recording.json")
            and recorded["wheel"]["sha256"] == packaged["sha256"]
            and recorded["wheel"]["native"] == packaged["native"],
            "Installed native capture or stack attribution does not identify the selected full workflow")
    check_record(demo, recorded["demo"])
    check_record(stocks["both"], recorded["stock_histogram"])
    saved = native / "histogram.npz"
    check_record(saved, recorded["histogram"])
    require(all(recorded["histogram"]["exact_stock_fields"].values()), "Profile rejected stock histogram fields")
    histograms(saved, stocks["both"], "both")
    verify_loaded(native, packaged)
    archive.add(saved, FINAL / "installed-histograms.npz")
    archive.members(native, FINAL / "native", ("recording.json", "library-maps.json", "machine.json",
                                                "command.json", "process.json"),
                    optional=("application.log", "perf.log", "provider-elf-sections.txt"))
    archive.members(decoded, FINAL / "native", ("summary.json", "owners.tsv", "self.tsv", "inclusive.tsv",
                                                 "within_owner_self.tsv", "within_owner_inclusive.tsv"),
                    optional=("header.txt", "buildids.txt", "postprocess.log"))
    optional = output / "native-unprofiled"
    if plan.get("unprofiled_overhead") or optional.is_dir():
        audit_file = optional / "diagnostic.json"
        audit = read(audit_file)
        require(audit["passed"] and audit["repeat"] == repetitions == 10 and audit["events_per_pass"] == EVENTS
                and same_path(audit["profile_recording"], native / "recording.json")
                and all(audit["stock_exact_fields"].values()) and all(audit["profiled_exact_fields"].values()),
                "Unprofiled diagnostic does not reproduce the selected native workload")
        command = read(optional / "command.json")
        profiled_argv = read(native / "command.json")
        profiled_argv = profiled_argv[profiled_argv.index("--") + 1:]
        process = read(optional / "application.json")
        unprofiled_argv = [*profiled_argv[:3], str(optional / "histogram.npz"), *profiled_argv[4:]]
        require(command == {"recorded_visible_argv": profiled_argv, "unprofiled_argv": unprofiled_argv}
                and profiled_argv[1:] == [str(demo), str(original), str(saved), "-n", str(repetitions)]
                and process["argv"] == ["taskset", "-c", "3", *unprofiled_argv]
                and process["exit"] == 0 and process["threads_seen"] == [1] and process["cpus_seen"] == ["3"]
                and process["source_mapping_observed"] and process["wall_s"] == audit["wall_s"]
                and same_path(audit["process_receipt"], optional / "application.json"),
                "Matched unprofiled command or process receipt disagrees with the installed profiled run")
        check_record(optional / "histogram.npz", {"sha256": audit["histogram_sha256"]}, check_path=False)
        histograms(optional / "histogram.npz", saved, "both")
        histograms(optional / "histogram.npz", stocks["both"], "both")
        require(verify_loaded(optional, packaged) == audit["physically_mapped_libraries"],
                "Unprofiled runtime map differs from the captured audit")
        archive.members(optional, FINAL / optional.name, ("diagnostic.json", "application.json", "application.log",
                                                            "library-maps.json", "command.json", "histogram.npz"))


def collect_python_profile(archive, output, installed, packaged, manifest, bundle, demo, original, dataset, stocks):
    directory = output / "python-profile"
    summary = read(directory / "summary.json")
    provenance = read(directory / "provenance.json")
    comparison = read(directory / "histogram-comparison.json")
    process = read(directory / "visible.json")
    profile = directory / "profile.pstats"
    histogram = directory / "histograms.npz"
    expected = [str(installed), "-m", "cProfile", "-o", str(profile), str(demo), str(original),
                str(histogram), "--events", str(EVENTS)]
    require(read(directory / "command.json") == expected
            and process["argv"] == ["taskset", "-c", "3", *expected]
            and process["exit"] == 0 and process["threads_seen"] == [1] and process["cpus_seen"] == ["3"]
            and process["source_mapping_observed"],
            "Python profile did not execute the exact installed visible command on the original file")
    require(provenance["scope"] == "both" and provenance["mode"] == "full"
            and provenance["events"] == EVENTS and provenance["passes"] == 1
            and provenance["cpu"] == 3 and provenance["threads"] == 1
            and provenance["original"] == dataset and provenance["wheel"] == packaged
            and provenance["selected_defaults"] == manifest["demo_defaults"],
            "Python profile receipt differs from the selected wheel, visible defaults or original source")
    check_record(demo, provenance["demo"])
    check_record(bundle, provenance["bundle"])
    check_record(stocks["both"], provenance["stock"]["histogram"])
    check_record(profile, summary["pstats"])
    check_record(histogram, comparison, check_path=False)
    actual_fields = histogram_comparison(stocks["both"], histogram, "both", EVENTS)
    require(summary["passed"] and summary["exact_stock_histogram"]
            and actual_fields == comparison["exact_fields"] and all(actual_fields.values())
            and summary["source_and_runtime_audit"] ==
            validate_run(directory, "both", EVENTS, dataset["tfile_uuid"], manifest, packaged),
            "Python profile histogram or runtime receipts differ from the exact installed workflow")
    # Raw pstats remains in WSL; its size and SHA-256 are in the archived summary.
    archive.members(directory, FINAL / directory.name, ("summary.json", "summary.txt", "provenance.json",
                                                         "command.json", "visible.json", "visible.log",
                                                         "library-maps.json", "histograms.npz",
                                                         "histogram-comparison.json"))


def collect_phases_and_order(archive, output, reference, installed, packaged, defaults, original, observer, uuid, stocks, demo):
    phases = output / "phases"
    provenance = read(phases / "provenance.json")
    phase_summary = read(phases / "summary.json")
    require(provenance["cpu"] == 3 and provenance["threads"] == 1 and provenance["events"] == EVENTS
            and provenance["selected_defaults"] == defaults and provenance["original"]["tfile_uuid"] == uuid
            and provenance["wheel"]["sha256"] == packaged["sha256"] and set(phase_summary) == set(SCOPES),
            "Phase receipt differs from the selected installed wheel or full source")
    check_record(demo, provenance["demo"])
    archive.members(phases, FINAL / "phases", ("summary.json", "provenance.json"))
    for scope in SCOPES:
        directory = phases / scope
        summary = read(directory / "summary.json")
        comparison = read(directory / "histogram-comparison.json")
        process = read(directory / "visible.json")
        require(process["exit"] == 0 and process["threads_seen"] == [1] and process["cpus_seen"] == ["3"]
                and process["source_mapping_observed"], f"Phase execution audit failed: {directory}")
        check_record(stocks[scope], provenance["stock"][scope]["histogram"])
        check_record(directory / "histograms.npz", comparison["actual"])
        require(summary == phase_summary[scope] and summary["exact_stock_histogram"] and all(comparison["exact_fields"].values()),
                f"Phase instrumentation is missing the exact stock guard for {scope}")
        histograms(directory / "histograms.npz", stocks[scope], scope)
        verify_loaded(directory, packaged)
        archive.members(directory, FINAL / "phases" / scope,
                        ("summary.json", "visible.json", "command.json", "phase-worker.json", "histograms.npz",
                         "histogram-comparison.json", "library-maps.json"), optional=("visible.log",))

    order = output / "chunk-order"
    summary = read(order / "summary.json")
    proof = read(order / "evaluator-order.json")
    require(summary["passed"] and summary["installation"]["wheel_sha256"] == packaged["sha256"]
            and proof["passed"] and not proof["bitwise_ordering_differences"]
            and proof["original_uuid"] == uuid and same_path(proof["original"], original)
            and same_path(proof["stock_observer"], observer)
            and proof["matched_production_settings"] == defaults
            and proof["installed_visible_source_sha256"] == digest(demo)
            and same_path(summary["evaluator_only"]["proof"], order / "evaluator-order.json"),
            "Installed chunk/order comparison does not cover the selected visible source")
    short = reference / "benchmark-pilot-combined-002/01-full/histograms.npz"
    check_record(short, summary["file_order_cli"]["stock_histogram"])
    require(summary["file_order_cli"]["histograms_exact"] and summary["file_order_cli"]["events"] == 32,
            "Physical chunk test does not cover the independently saved stock interval")
    for step in summary["file_order_cli"]["chunk_sizes"]:
        histogram = order / f"chunk-{step}.npz"
        histograms(histogram, short, "both", 32)
        archive.members(order, FINAL / order.name, (histogram.name, *(f"chunk-{step}.{ext}" for ext in ("argv.json", "exit", "log"))))
    check_map(order / "evaluator-native-libraries.json", packaged["native"], installed)
    archive.members(order, FINAL / order.name, ("summary.json", "evaluator-order.json", "evaluator-native-libraries.json"),
                    optional=("evaluator-order.argv.json", "evaluator-order.exit", "evaluator-order.log"))


def collect_source(archive, output, reference, dataset, packaged, paths, specification):
    observer = reference / "full-clean-001"
    original = Path(dataset["path"]).resolve(strict=True)
    coverage = read(observer / "verification.json")
    require(dataset["entry_interval"] == [0, EVENTS] and original.stat().st_size == dataset["bytes"]
            and coverage["source_file_uuid"] == dataset["tfile_uuid"] and coverage["events"] == EVENTS
            and coverage["observer_branches"] == 7189
            and (observer / "exit-code.txt").read_text().strip() == "0"
            and checksum_records(observer / "input.sha256").get(original) == dataset["sha256"],
            "The archived original source and independent stock observer differ")
    archive.add(ROOT / "config/jets-input.json", FINAL / "source/jets-input.json")
    archive.add(specification, FINAL / "source/qualification-specification.json")
    archive.members(observer, FINAL / "source/stock-observer",
                    ("verification.json", "input.sha256", "reference.sha256", "exit-code.txt"),
                    optional=("process-status.txt", "command.txt", "initialized-tools.tsv"))
    archive.add(Path(packaged["data"]["provenance.json"]["path"]), FINAL / "source/calibration-provenance.json")
    archive.add(Path(packaged["workflow"]["data/payload-index.json"]["path"]), FINAL / "source/payload-index.json")
    archive.add(paths["partition"] / "manifest.json", FINAL / "source/partition.json")
    archive.add(paths["installation"] / "installations.json", FINAL / "source/installations.json")
    stock_dirs = {"both": "combined", "small": "small", "large": "large"}
    stocks = {}
    for scope, name in stock_dirs.items():
        stock_dir = reference / f"full-file-pilot-{name}-002"
        stock = stock_endpoint(stock_dir, scope, original, dataset, digest)
        stocks[scope] = Path(stock["histogram"]["path"])
        prefix = FINAL / "source" / f"stock-{scope}"
        archive.add(stocks[scope], prefix / "histograms.npz")
        archive.add(stock_dir / "provenance.json", prefix / "provenance.json")
        for role, item in stock["legs"].items():
            archive.add(Path(item["path"]), prefix / f"{role}.json")
    # This is the compact independently prepared input to the file-order test.
    short = reference / "benchmark-pilot-combined-002/01-full/histograms.npz"
    archive.add(short, FINAL / "source/stock-both-32/histograms.npz")
    archive.add(short.parents[1] / "provenance.json", FINAL / "source/stock-both-32/provenance.json")
    if (output / "scoped-acceptance.json").is_file():
        archive.add(output / "scoped-acceptance.json", FINAL / "scoped-acceptance.json")
    return original, observer / "reference.root", stocks


def collect_controller(archive, controller, specification, output):
    require((controller / "exit-code").read_text().strip() == "0",
            "The supplied shipment controller did not complete")
    resolved = read(controller / "resolved.json")
    require(resolved["specification_sha256"] == digest(specification) and same_path(resolved["output"], output),
            "Controller belongs to another qualification specification")
    archive.members(controller, FINAL / "controller",
                    ("specification.json", "specification.sha256", "resolved.json", "stages.tsv", "exit-code",
                     "started.utc", "ended.utc"), optional=("controller.sha256",))
    check_record(controller / "specification.json", {"sha256": digest(specification)}, check_path=False)
    for stage in sorted(controller.glob("[0-9][0-9]*-*")):
        if stage.is_dir():
            require((stage / "exit-code").read_text().strip() == "0", f"Controller stage did not complete: {stage}")
            archive.members(stage, FINAL / "controller" / stage.name,
                            ("command.txt", "cpu", "started.utc", "ended.utc", "exit-code", "stage.log"))


def main(specification: Path, report: Path = REPORT, controller: Path | None = None):
    """Copy verified shipment artifacts and receipts into delivery and evidence/final."""
    specification = specification.resolve(strict=True)
    report = report.resolve(strict=True)
    require(report.is_dir(), f"Expected an existing campaign report directory: {report}")
    plan = read(specification)
    paths = {key: (ROOT / plan[key]).resolve(strict=True) for key in
             ("output", "bundle", "installation", "provider", "payload", "sdk", "overlay", "partition")}
    output = paths["output"]
    bundle = paths["bundle"]
    require(not report.is_relative_to(output) and not output.is_relative_to(report),
            "Keep the qualification and Windows archive directory trees separate")
    bundle_path = bundle / "bundle.json"
    manifest = read(bundle_path)
    wheel = archive_member(bundle, manifest["wheel"]["name"])
    demo = archive_member(bundle, manifest["visible_script"]["name"])
    require(demo.name == "jets_demo.py" and wheel.suffix == ".whl", "Unexpected wheel or visible script name")
    python = paths["installation"] / "pip/env/bin/python"
    packaged, manifest = checked_installation(python, demo, wheel, bundle_path, digest, wheel_identity)
    for key in ("provider", "payload"):
        check_record(paths[key], manifest[key], check_path=False)
    defaults = manifest["demo_defaults"]
    dataset = read(ROOT / "config/jets-input.json")
    reference = ROOT / "results/reference"
    archive = Archive(report)
    archive.members(bundle, Path("delivery"), (wheel.name, demo.name, "bundle.json"), optional=("SHA256SUMS",))
    original, observer, stocks = collect_source(archive, output, reference, dataset, packaged, paths, specification)
    uuid = dataset["tfile_uuid"]
    collect_direct(archive, output, packaged, defaults, paths["partition"], observer, original, uuid)
    collect_framework(archive, output, paths, defaults, observer, original, uuid, stocks)
    collect_shipped(archive, output, paths["installation"], packaged, defaults, original, stocks, digest(demo))
    collect_profile(archive, output, plan, packaged, defaults, original, demo, stocks)
    collect_python_profile(archive, output, python, packaged, manifest, bundle_path, demo, original, dataset, stocks)
    collect_phases_and_order(archive, output, reference, paths["installation"], packaged, defaults,
                             original, observer, uuid, stocks, demo)
    if controller is not None:
        collect_controller(archive, controller.resolve(strict=True), specification, output)
    count, manifest_path = archive.publish(specification, output, wheel)
    typer.echo(f"Verified or archived {count} files. Manifest at {manifest_path}")


if __name__ == "__main__":
    typer.run(main)
