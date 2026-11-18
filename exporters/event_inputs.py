"""Describe collection context, selections and event weight for the frozen jet YAML."""

def export_event_inputs():
    real, integer = "real", "int"
    p4 = [(x,real) for x in ("pt","eta","phi","m")]
    event = [("event",integer)]
    stages = [
        ("jet_event_inputs","event_inputs",event+[("pv_index",integer),("pv_index_present",integer),
          ("npv",real),("npv_present",integer)],5,{"primary":1,"pileup":3,"dummy":0,"min_tracks":2}),
        ("jet_pileup_truth","pileup_truth",p4+event,2,
         {"hs_dr":0.3,"pu_dr":0.6,"hs_pt":10000.0,"pu_pt":0.0}),
        ("large_selection","large_selection",p4,3,
         {"min_pt":200000.0,"max_pt":3000000.0,"max_rapidity":2.0,"min_mass":40000.0,"max_mass":600000.0}),
        ("jet_event_sf","event_sf",event,0,{}),
    ]
    corrections=[]
    for name,node,inputs,size,config in stages:
        output={"name":"result","type":"real"}
        data={"nodetype":"extension","type":"org.atlas.jets."+node,"version":1,"config":config}
        if size:
            data["output_shape"]=[size]
            output["shape"]=[size]
        corrections.append({"name":name,"version":1,"inputs":[{"name":k,"type":t} for k,t in inputs],
                            "output":output,"data":data})
    return corrections, {
        "release":"6ce3f7ae7c41dd9eaec170c3c349df15faef7e5c",
        "vertex":"JetCalibrationTool::initializeEvent, JetUncertaintiesTool::getDefaultEventInfo, JetVertexNNTagger::findHSVertex",
        "truth":"JetPileupLabelingTool::decorate and JetPileupLabelingTool.h defaults for both configured collections",
        "selection":"AsgSelectionAlg, AsgPtEtaSelectionTool, AsgMassSelectionTool with config/jets-a.yaml thresholds",
        "event_sf":"AsgEventScaleFactorAlg with valid per-jet SF or neutral weight in source order"}
