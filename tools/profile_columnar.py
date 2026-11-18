"""Time existing direct-workflow call boundaries without changing the numerical calls."""
import collections
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from columnar import jets
import typer

observations=collections.defaultdict(lambda:{"calls":0,"rows":0,"wall_ns":0,"cpu_ns":0})
def timed(name,call,args,kwargs,rows=0):
    wall,cpu=time.perf_counter_ns(),time.process_time_ns()
    try:
        return call(*args,**kwargs)
    finally:
        record=observations[name]
        record["wall_ns"]+=time.perf_counter_ns()-wall
        record["cpu_ns"]+=time.process_time_ns()-cpu
        record["calls"]+=1
        record["rows"]+=rows

class Tool:
    def __init__(self,name,original):
        self.name,self.original=name,original
    def evaluate(self,*args,**kwargs):
        size=next((arg.shape[0] for arg in args if hasattr(arg,"shape") and arg.ndim),1)
        return timed(self.name,self.original.evaluate,args,kwargs,int(size))

class NodeSet:
    def __init__(self,original):
        self.original,self.cached=original,{}
    def __getitem__(self,name):
        if name not in self.cached:
            self.cached[name]=Tool(name,self.original[name])
        return self.cached[name]

def main():
    report=Path(os.environ["JETS_PHASE_OUTPUT"])
    if report.exists():
        raise FileExistsError(report)
    def wrap(obj,name,label):
        call=getattr(obj,name)
        def wrapped(*args,**kwargs):
            return timed(label,call,args,kwargs)
        setattr(obj,name,wrapped)
    original=jets.load_corrections
    def load(*args,**kwargs):
        nodes,buffers=timed("load_provider_and_payload",original,args,kwargs)
        return NodeSet(nodes),buffers
    jets.load_corrections=load
    wrap(jets.uproot.behaviors.TBranch.HasBranches,"arrays","input_tree_arrays")
    wrap(jets.hist.Hist,"fill","histogram_fill")
    wrap(jets.np,"savez","write_npz")
    wall,cpu=time.perf_counter_ns(),time.process_time_ns()
    try:
        typer.run(jets.main)
    finally:
        with report.open("x") as target:
            json.dump({"argv":sys.argv,"process":jets.check_execution(),
                       "main_wall_ns":time.perf_counter_ns()-wall,"main_cpu_ns":time.process_time_ns()-cpu,
                       "observations":dict(observations)},target,indent=2)
            target.write("\n")

if __name__=="__main__":
    main()
