"""Check numerical JSON loading on the actual EtaJES constants in this campaign."""

import json
import struct
from pathlib import Path

import correctionlib
import typer


def main(payload: Path, output: Path):
    data=json.loads(payload.read_text())
    constants={}
    for corr in data["corrections"]:
        if not corr["name"].endswith("eta_jes"):
            continue
        cfg=corr["data"]["config"]
        for i,value in enumerate(cfg["eta_edges"]):
            constants[f"{corr['name']}_edge_{i}"]=value
        for i,row in enumerate(cfg["bins"]):
            for j,value in enumerate(row["eta_coefficients"]):
                constants[f"{corr['name']}_eta_{i}_{j}"]=value
    edges = [{"name":k, "version":1, "inputs":[], "output":{"name":"y","type":"real"}, "data":v}
             for k,v in constants.items()]
    fixture=json.dumps({"schema_version":2,"corrections":edges}, separators=(",",":"))
    cset=correctionlib.CorrectionSet.from_string(fixture)
    errors=[]
    for k,v in constants.items():
        result=cset[k].evaluate()
        if struct.pack(">d", result)!=struct.pack(">d",v):
            errors.append({"name":k,"json":json.dumps(v),"expected":v.hex(),"observed":result.hex()})
    with output.open("x") as f:
        json.dump({"core":correctionlib.__version__, "evaluated":len(constants),"errors":len(errors),"examples":errors[:10]},f,indent=2)


if __name__=="__main__":
    typer.run(main)
