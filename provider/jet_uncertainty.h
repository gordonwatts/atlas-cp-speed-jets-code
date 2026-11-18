#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string>

namespace correction {
class PluginConfig;
class PluginRegistry;
}

namespace jets {

// The supplied values are the nominally calibrated xAOD auxiliaries.
struct JetUncertaintyInput {
  std::array<double, 4> p4;
  float average_mu;
  float uncertainty_npv;
  bool event_available;
  bool preselected;
  std::optional<int> parton_truth_label;
  std::optional<int> large_r_truth_label;
  std::optional<std::array<double, 4>> calo_scale;
};

enum class JetUncertaintyStatus { error = 0, out_of_validity = 1, ok = 2 };

struct JetUncertaintyResult {
  std::array<double, 4> p4;
  std::optional<JetUncertaintyStatus> status;
  std::optional<bool> is_bjet;
};

class JetUncertaintyModel {
 public:
  explicit JetUncertaintyModel(const correction::PluginConfig& configuration);
  ~JetUncertaintyModel();

  JetUncertaintyResult evaluate(const JetUncertaintyInput& input, const std::string& systematic) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> model_;
};

}  // namespace jets

void register_jet_uncertainty(correction::PluginRegistry& registry);
