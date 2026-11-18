#pragma once

#include "jet_histogram.h"
#include "truth_context.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>


namespace correction {
class PluginRegistry;
}

namespace jets {

// Same values as CP::CorrectionCode in AnalysisBase 25.2.117.
enum class JmrStatus { error = 0, out_of_validity = 1, ok = 2 };
enum class JmrBinning { pt_abs_mass, energy_log_mass_abs_eta };

struct LargeJmrSettings {
  float max_abs_eta;
  float max_mass_mev;
  float max_pt_mev;
  float mev_to_gev;
  double max_match_delta_r;
  double axis_edge_offset;
  double minimum_shift;
  double nominal_scale;
  double nominal_resolution;
  double zero_response_fallback;
  std::string fallback_topology;
  std::vector<std::string> topology_by_label;
};

struct LargeJmrVariation {
  JmrBinning binning;
  std::string_view topology_filter;
  const CalibrationHistogram& histogram;
  float parameter;
};

struct LargeJmrResult {
  P4 p4;
  JmrStatus status;
  bool write_p4;  // Apply the xAOD float writeback only when true.
};

const P4* nearest_truth_jet(const P4& reco, TruthJetRange truth, double maximum_delta_r);

LargeJmrResult apply_large_jmr(const P4& reco, std::optional<int> truth_label,
                               TruthJetRange truth, const CalibrationHistogram& response,
                               const LargeJmrVariation& variation,
                               const LargeJmrSettings& settings);

}  // namespace jets

void register_large_jmr(correction::PluginRegistry& registry);
