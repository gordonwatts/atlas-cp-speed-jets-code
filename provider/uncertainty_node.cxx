#include "jet_uncertainty.h"
#include <correction_plugin.h>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

using Inputs = std::vector<correction::Variable::Type>;
using VarType = correction::Variable::VarType;
constexpr std::size_t uncertainty_output_count = 8;

struct InputIndex {
  std::array<std::size_t, 4> p4;
  std::size_t average_mu;
  std::size_t uncertainty_npv;
  std::size_t event_available;
  std::size_t preselected;
  std::size_t parton_truth_label;
  std::size_t parton_truth_label_available;
  std::size_t large_r_truth_label;
  std::size_t large_r_truth_label_available;
  std::array<std::size_t, 4> calo_p4;
  std::size_t calo_available;
  std::size_t systematic;
};

std::array<double, 4> read_p4(const Inputs& values, const std::array<std::size_t, 4>& index) {
  return {std::get<double>(values.at(index[0])), std::get<double>(values.at(index[1])),
          std::get<double>(values.at(index[2])), std::get<double>(values.at(index[3]))};
}

bool read_flag(const Inputs& values, std::size_t index) {
  const auto value = std::get<std::int64_t>(values.at(index));
  if (value != 0 && value != 1) throw std::invalid_argument("Jet uncertainty flag must be zero or one");
  return value != 0;
}

std::optional<int> read_truth_label(const Inputs& values, std::size_t index, std::size_t availability) {
  if (!read_flag(values, availability)) return std::nullopt;
  const auto label = std::get<std::int64_t>(values.at(index));
  if (label < std::numeric_limits<int>::min() || label > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("Jet uncertainty truth label lies outside the xAOD int range");
  }
  return static_cast<int>(label);
}

class JetUncertaintyNode final : public correction::PluginNode {
 public:
  JetUncertaintyNode(InputIndex indices, const correction::PluginConfig& configuration)
      : indices_(indices), model_(configuration) {}

  std::size_t output_size() const override { return uncertainty_output_count; }

  double evaluate(const Inputs&) const override {
    throw std::logic_error("Jet uncertainty has an array output");
  }

  void evaluate_array(const Inputs& values, double* output, std::size_t size) const override {
    if (!output || size < output_size()) throw std::invalid_argument("Jet uncertainty output is too small");
    jets::JetUncertaintyInput input{
        read_p4(values, indices_.p4),
        static_cast<float>(std::get<double>(values.at(indices_.average_mu))),
        static_cast<float>(std::get<double>(values.at(indices_.uncertainty_npv))),
        read_flag(values, indices_.event_available),
        read_flag(values, indices_.preselected),
        read_truth_label(values, indices_.parton_truth_label, indices_.parton_truth_label_available),
        read_truth_label(values, indices_.large_r_truth_label, indices_.large_r_truth_label_available),
        std::nullopt};
    if (read_flag(values, indices_.calo_available)) input.calo_scale = read_p4(values, indices_.calo_p4);
    const auto& systematic = std::get<std::string>(values.at(indices_.systematic));
    const auto result = model_.evaluate(input, systematic);
    std::copy(result.p4.begin(), result.p4.end(), output);
    output[4] = result.status ? static_cast<double>(*result.status) : std::numeric_limits<double>::quiet_NaN();
    output[5] = result.status.has_value() ? 1.0 : 0.0;
    output[6] = result.is_bjet ? static_cast<double>(*result.is_bjet) : std::numeric_limits<double>::quiet_NaN();
    output[7] = result.is_bjet.has_value() ? 1.0 : 0.0;
  }

 private:
  InputIndex indices_;
  jets::JetUncertaintyModel model_;
};

}  // namespace

void register_jet_uncertainty(correction::PluginRegistry& registry) {
  registry.register_node("org.atlas.jets.uncertainty", 1,
      [](const correction::PluginConfig& config, const correction::PluginContext& context) {
    if (context.output_size() != uncertainty_output_count || context.child_count()) {
      throw std::invalid_argument("Jet uncertainty requires eight output values and no children");
    }
    const InputIndex indices{
        {context.input_index("pt", VarType::real), context.input_index("eta", VarType::real),
         context.input_index("phi", VarType::real), context.input_index("m", VarType::real)},
        context.input_index("average_mu", VarType::real),
        context.input_index("uncertainty_npv", VarType::real),
        context.input_index("event_available", VarType::integer),
        context.input_index("preselected", VarType::integer),
        context.input_index("parton_truth_label", VarType::integer),
        context.input_index("parton_truth_label_available", VarType::integer),
        context.input_index("large_r_truth_label", VarType::integer),
        context.input_index("large_r_truth_label_available", VarType::integer),
        {context.input_index("calo_pt", VarType::real), context.input_index("calo_eta", VarType::real),
         context.input_index("calo_phi", VarType::real), context.input_index("calo_m", VarType::real)},
        context.input_index("calo_available", VarType::integer),
        context.input_index("systematic", VarType::string)};
    return std::make_shared<JetUncertaintyNode>(indices, config);
  });
}
