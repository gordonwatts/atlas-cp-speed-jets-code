#ifndef ATLAS_JETS_LARGE_JMS_H
#define ATLAS_JETS_LARGE_JMS_H

#include "jet_common.h"
#include "jet_histogram.h"

namespace correction {
class PluginRegistry;
}

namespace jets {

struct LargeJmsSettings {
  double minimum_energy_gev;
  double axis_edge_offset;
  double mev_per_gev;
  bool keep_pt_fixed;
};

// The input is the float-written EtaJES scale. The result is the float-written
// JMS scale. The lookup uses the original float DetectorEta decoration.
P4 apply_large_jms(const P4& eta_jes, float detector_eta,
                   const CalibrationHistogram& response, const LargeJmsSettings& settings);

}  // namespace jets

void register_large_jms(correction::PluginRegistry& registry);

#endif
