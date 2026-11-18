#pragma once

#include <JetUncertainties/UncertaintyGroup.h>

#include <memory>
#include <string>
#include <vector>

class TEnv;
class TFile;

namespace jets {

struct UncertaintyFlavourInput {
  TFile& file;
  const std::string& fraction_key;
  const std::string& error_key;
  bool abs_eta;
};

std::vector<std::unique_ptr<jet::UncertaintyGroup>> initialize_uncertainty_groups(
    TEnv& settings, TFile& histograms, const UncertaintyFlavourInput& flavour,
    const std::string& jet_definition, const std::string& mc_type,
    const std::string& group_prefix, float energy_scale);

}  // namespace jets
