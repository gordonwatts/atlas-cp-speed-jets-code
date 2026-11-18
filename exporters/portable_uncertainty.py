"""Resolve the active JetUncertainties configuration to a dependency-free evaluation plan."""

import json
import re
from collections import defaultdict
from pathlib import Path

import typer


# ParticleJetTools/LargeRJetLabelEnum.h, MC23 recommendations.
TRUTH = dict(tqqb=1, wqq=2, zqq=5, wqq_from_t=6, other_from_t=7,
             other_from_v=8, notruth=9, qcd=10)
MODES = {"true": "Full", "false": "None", "none": "None", "onlyx": "OnlyX"}
SPECIAL = ("pileup_offsetmu", "pileup_offsetnpv", "pileup_ptterm", "pileup_rhotopology",
           "flavor_composition", "flavor_response", "bjes_response", "larger_topologyuncertainty")


def split(value):
    return [name for name in re.split(r"[,\s]+", value.strip()) if name]


def resolve_plan(model):
    settings = model["settings"]
    jet = model["jet_definition"]
    mc = model["mc_type"]
    group_prefix = model["group_prefix"]
    def key(name):
        return name.replace("MCTYPE", mc) + "_" + jet

    def fields(family):
        rows = defaultdict(dict)
        for setting, value in settings.items():
            match = re.fullmatch(family + r"\.(\d+)\.(\w+)", setting)
            if match:
                rows[int(match[1])][match[2]] = value.strip().replace("MCTYPE", mc)
        return [rows[n] for n in sorted(rows)]

    def prefixed(name):
        if name.startswith("NOPREFIX_"):
            return name.removeprefix("NOPREFIX_")
        if name.lower().startswith(group_prefix.lower()):
            return group_prefix + name[len(group_prefix):]
        return group_prefix + name

    groups = {}
    numbers = {}
    def new_group(row):
        name = prefixed(row["Name"])
        if name in groups or int(row.get("SubGroup", "0")):
            raise ValueError(f"Unsupported repeated group or subgroup: {name}")
        group = {"correlation": row.get("Corr", ""), "components": []}
        groups[name] = group
        return group

    for row in fields("JESGroup"):
        n = int(row["Group"])
        if n in numbers:
            raise ValueError(f"Repeated group number {n}")
        numbers[n] = new_group(row)
    for row in fields("JESComponent"):
        number = int(row.get("Group", "0"))
        group = numbers[number] if number else new_group(row)
        if row.get("Split", "0") != "0" or row.get("Topology") or row.get("CombMassType"):
            raise ValueError(f"Unsupported component option {row}")
        sub = split(row.get("SubComp", ""))
        sources = [split(row.get("Hists") or row["Name"])] if not sub else [[s] for s in sub]
        for names in sources:
            histograms = [key(n) for n in names]
            special = row.get("Special", "").lower() in ("true", "yes")
            kind = next((kind for kind in SPECIAL if row["Name"].lower().startswith(kind)), None) if special else "table"
            param = row["Param"]
            labels = [TRUTH[name.lower()] for name in split(row.get("LargeRJetTruthLabels", ""))]
            if labels and row.get("LargeRJetTruthLabelName") != "R10TruthLabel_R21Precision_2022v1":
                raise ValueError("Unsupported truth-label decoration")
            if not kind or param not in ("Pt", "PtEta", "PtAbsEta", "PtAbsMass", "eLOGmOeAbsEta"):
                raise ValueError(f"Unsupported component {row}")
            for table in histograms:
                if table not in model["histograms"]:
                    raise KeyError(table)
            group["components"].append({
                "kind":kind, "param":param, "scale":row.get("Scale","FourVec").lower(),
                "mode": MODES[row.get("Interp","true").lower()],
                "calo": row.get("MassDef","").lower()=="calo", "labels":labels,
                "tables":histograms, "validity":key(row["VHist"]) if row.get("VHist") else "",
            })
    for name, group in groups.items():
        if not group["components"] or len(group["components"])>1 and group["correlation"] not in ("Correlated","Uncorrelated"):
            raise ValueError(f"Unsupported group composition {name}: {group}")
    bases = {s["basename"] for s in model["states"].values()}
    if not bases <= groups.keys():
        raise ValueError(f"Missing initialized group(s): {bases-groups.keys()}")

    resolution = {}
    for setting, value in settings.items():
        if setting.startswith("Nominal"):
            name, param, mode = (s.strip() for s in value.split(","))
            if param != "PtAbsEta":
                raise ValueError(value)
            resolution[setting] = {"table":key(name), "mode":MODES[mode.lower()]}
    if settings.get("FileValidHistogram") and (settings["FileValidHistParam"], settings["FileValidHistMassDef"]) != ("PtAbsMassEta","Calo"):
        raise ValueError("Unexpected file validity parametrization")
    return {"groups":groups, "resolutions":resolution,
            "file_validity":key(settings["FileValidHistogram"]) if settings.get("FileValidHistogram") else "",
            "ref_npv":float(settings.get("Pileup.NPVRef",-1)),
            "ref_mu":float(settings.get("Pileup.MuRef",-1)),
            "smear_only_mc":settings["ResolutionSmearOnlyMC"].lower()=="true"}


def main(source: Path, output: Path, inventory: Path):
    if output.exists() or inventory.exists():
        raise FileExistsError("Use a fresh payload and plan inventory")
    data=json.loads(source.read_bytes())
    plans={}
    for correction in data["corrections"]:
        if correction["name"] in ("small_uncertainty","large_uncertainty"):
            cfg=correction["data"]["config"]
            plan=resolve_plan(cfg)
            plans[correction["name"]]=plan
            cfg["portable_plan"]=plan
    if len(plans)!=2:
        raise ValueError("Expected both jet uncertainty configurations")
    with output.open("x") as f:
        json.dump(data,f,separators=(",",":"),allow_nan=False)
    with inventory.open("x") as f:
        json.dump(plans,f,indent=2)
        f.write("\n")


if __name__=="__main__":
    typer.run(main)
