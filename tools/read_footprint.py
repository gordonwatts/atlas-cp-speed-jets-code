"""Reconcile the direct input branch plan to distinct on-disk ROOT leaves and basket bytes."""
import json
from pathlib import Path
import sys
CHECKOUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CHECKOUT))
from columnar.inputs import branch_plan, open_root, resolve_branches
import typer


def inventory(tree, requested):
    leaves=set()
    tables={}
    def inspect(branch):
        name=branch.name
        if name in tables:
            return
        children=list(branch.branches)
        if children:
            for child in children:
                inspect(child)
            return
        raw_leaves=branch.member("fLeaves")
        names=[str(leaf.member("fName")) for leaf in raw_leaves]
        for leaf_name in names:
            leaves.add((name,leaf_name))
        tables[name]={"leaves":names,"baskets":branch.num_baskets,
                      "compressed_basket_bytes":sum(branch.basket_compressed_bytes(i) for i in range(branch.num_baskets)),
                      "uncompressed_basket_bytes":sum(branch.basket_uncompressed_bytes(i) for i in range(branch.num_baskets))}
        count=branch.count_branch
        if count is not None:
            inspect(count)

    for key in requested:
        inspect(tree[key])
    compressed=sum(v["compressed_basket_bytes"] for v in tables.values())
    raw=sum(v["uncompressed_basket_bytes"] for v in tables.values())
    return {"requested":requested,"branch_count":len(requested),"physical_basket_branches":len(tables),
            "distinct_leaves":len(leaves),"compressed_bytes":compressed,"uncompressed_bytes":raw,
            "compressed_kB_per_event":compressed/tree.num_entries/1000,
            "uncompressed_kB_per_event":raw/tree.num_entries/1000,"inventory":tables}


def main(original:Path,output:Path):
    with open_root(original) as root:
        tree=root["CollectionTree"]
        data={"original":str(original),"root_uuid":str(root.file.uuid),"events":tree.num_entries,"tree":"CollectionTree",
              "basis":"One basket copy per selected terminal branch, selected for the full entry range; both byte totals include ROOT basket headers. Cached, repeated and physical disk I/O are outside this measure.",
              "scopes":{}}
        for label,selected in (("small",("sr",)),("large",("lr",)),("combined",("sr","lr"))):
            data["scopes"][label]={}
            for mode in ("stored","nominal","full"):
                actual=resolve_branches(tree,branch_plan(selected,mode))
                data["scopes"][label][mode]=inventory(tree,sorted({v for v in actual.values() if v is not None}))
    with output.open("x") as store:
        json.dump(data,store,indent=2)
        store.write("\n")

if __name__=="__main__":
    typer.run(main)
