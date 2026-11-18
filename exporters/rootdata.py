"""Offline serialization of numeric ROOT histograms consumed by both jet recipes."""

import math
import re


def export_histogram(hist):
    """Keep the ROOT class, variable/fixed axes, global bin order and raw Sumw2."""
    if not hist or not re.fullmatch(r"TH[123][CSIFD]", hist.ClassName()):
        raise ValueError(f"unsupported histogram: {hist}")
    import ROOT

    axes = []
    for axis in (hist.GetXaxis(), hist.GetYaxis(), hist.GetZaxis())[: hist.GetDimension()]:
        bins = axis.GetXbins()
        axes.append({
            "nbins": axis.GetNbins(),
            "min": axis.GetXmin(),
            "max": axis.GetXmax(),
            "edges": [bins.At(i) for i in range(bins.GetSize())],
            "range": [int(axis.GetFirst()), int(axis.GetLast())] if axis.TestBit(ROOT.TAxis.kAxisRange) else None,
        })
    # GetBinContent(i) uses ROOT global bins with X varying fastest, including flows.
    contents = [hist.GetBinContent(i) for i in range(hist.GetNcells())]
    variances = [hist.GetSumw2().At(i) for i in range(hist.GetSumw2N())]
    if not all(map(math.isfinite, contents)) or not all(map(math.isfinite, variances)):
        raise ValueError(f"non-finite source bin in {hist.GetName()}")
    return {"class": hist.ClassName(), "axes": axes, "content": contents, "sumw2": variances}
