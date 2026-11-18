#include "large_jms.h"

#include <correction_plugin.h>
#include <rapidjson/document.h>

#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace {

double high_edge(const jets::CalibrationAxis& axis) {
  return axis.GetBinLowEdge(axis.GetNbins() + 1);
}

double clamp_axis(double value, const jets::CalibrationAxis& axis, double offset) {
  const double low = axis.GetBinLowEdge(1);
  const double high = high_edge(axis);
  if (value >= high) value = high - offset;
  if (value <= low) value = low + offset;
  return value;
}

float mass_response(const jets::CalibrationHistogram& histogram, double energy, double log_mass_over_energy,
                    double abs_detector_eta, const jets::LargeJmsSettings& settings) {
  const jets::CalibrationAxis& energy_axis = *histogram.GetXaxis();
  const double high = high_edge(energy_axis);
  if (energy >= high) energy = high - settings.axis_edge_offset;
  if (energy <= settings.minimum_energy_gev) return 1.0f;
  const double low = energy_axis.GetBinLowEdge(1);
  if (energy <= low) energy = low + settings.axis_edge_offset;

  if (std::isnan(log_mass_over_energy)) return 1.0f;
  const double log_mass = clamp_axis(log_mass_over_energy, *histogram.GetYaxis(),
                                     settings.axis_edge_offset);
  const double eta = clamp_axis(abs_detector_eta, *histogram.GetZaxis(),
                                settings.axis_edge_offset);
  return static_cast<float>(jets::calib_interpolate(histogram, energy, log_mass, eta));
}

class LargeJmsNode final : public correction::PluginNode {
 public:
  LargeJmsNode(std::array<size_t, 5> indices, std::unique_ptr<const jets::CalibrationHistogram> response,
               jets::LargeJmsSettings settings)
      : indices_(indices), response_(std::move(response)), settings_(settings) {}

  size_t output_size() const override { return 4; }

  double evaluate(const std::vector<correction::Variable::Type>&) const override {
    throw std::logic_error("Large-R JMS returns four coordinates");
  }

  void evaluate_array(const std::vector<correction::Variable::Type>& values,
                      double* output, size_t size) const override {
    if (!output || size < output_size()) {
      throw std::invalid_argument("Large-R JMS output needs four coordinates");
    }

    const jets::P4 eta_jes(std::get<double>(values.at(indices_[0])),
                          std::get<double>(values.at(indices_[1])),
                          std::get<double>(values.at(indices_[2])),
                          std::get<double>(values.at(indices_[3])));
    const float detector_eta = static_cast<float>(std::get<double>(values.at(indices_[4])));
    const jets::P4 calibrated = jets::apply_large_jms(eta_jes, detector_eta, *response_, settings_);
    output[0] = calibrated.Pt();
    output[1] = calibrated.Eta();
    output[2] = calibrated.Phi();
    output[3] = calibrated.M();
  }

 private:
  std::array<size_t, 5> indices_;
  std::unique_ptr<const jets::CalibrationHistogram> response_;
  jets::LargeJmsSettings settings_;
};

}  // namespace

namespace jets {

P4 apply_large_jms(const P4& eta_jes, float detector_eta,
                   const jets::CalibrationHistogram& response, const LargeJmsSettings& settings) {
  const double abs_detector_eta = std::abs(static_cast<double>(detector_eta));
  const double mass = eta_jes.M();
  float corrected_mass = static_cast<float>(mass);
  double corrected_pt = eta_jes.Pt();

  if (abs_detector_eta < high_edge(*response.GetZaxis()) && mass != 0.0) {
    const double energy = eta_jes.E();
    const double mass_over_energy = mass / energy;
    double factor = 1.0;
    if (mass_over_energy > 0.0) {
      factor = mass_response(response, energy / settings.mev_per_gev,
                             std::log(mass_over_energy), abs_detector_eta, settings);
    }
    if (factor == 0.0) factor = 1.0;
    corrected_mass = static_cast<float>(mass / factor);
    if (corrected_mass > energy) corrected_mass = static_cast<float>(mass);

    if (!settings.keep_pt_fixed) {
      // corrected_mass * corrected_mass is evaluated in float in the release.
      corrected_pt = std::sqrt(energy * energy - corrected_mass * corrected_mass) /
                     std::cosh(eta_jes.Eta());
    }
  }

  CartesianP4 calo;
  calo.SetPtEtaPhiM(corrected_pt, eta_jes.Eta(), eta_jes.Phi(), corrected_mass);
  P4 result = eta_jes;
  result.SetPxPyPzE(calo.Px(), calo.Py(), calo.Pz(), calo.E());
  return stored_p4(result);
}

}  // namespace jets

void register_large_jms(correction::PluginRegistry& registry) {
  registry.register_node("org.atlas.jets.large_jms", 1,
      [](const correction::PluginConfig& config, const correction::PluginContext& context) {
    config.require_keys({"histogram", "binning", "minimum_energy_gev", "axis_edge_offset",
                         "mev_per_gev", "keep_pt_fixed"});
    if (context.output_size() != 4 || context.child_count() != 0) {
      throw std::invalid_argument("Large-R JMS requires four output coordinates and no children");
    }
    if (config.string("binning") != "e_LOGmOe_eta") {
      throw std::invalid_argument("Large-R JMS v1 requires e_LOGmOe_eta binning");
    }

    rapidjson::Document document;
    document.Parse<rapidjson::kParseFullPrecisionFlag>(config.json().c_str());
    if (document.HasParseError() || !document.IsObject() || !document["keep_pt_fixed"].IsBool()) {
      throw std::invalid_argument("Large-R JMS requires a boolean keep_pt_fixed");
    }
    const jets::LargeJmsSettings settings{
        config.real("minimum_energy_gev"), config.real("axis_edge_offset"),
        config.real("mev_per_gev"), document["keep_pt_fixed"].GetBool()};
    if (!std::isfinite(settings.minimum_energy_gev) || settings.minimum_energy_gev < 0.0 ||
        !std::isfinite(settings.axis_edge_offset) || settings.axis_edge_offset <= 0.0 ||
        !std::isfinite(settings.mev_per_gev) || settings.mev_per_gev <= 0.0) {
      throw std::invalid_argument("Invalid Large-R JMS energy or axis settings");
    }

    auto response = jets::restore_calibration_histogram(document["histogram"]);
    if (!jets::is_th3f(*response)) throw std::invalid_argument("Large-R JMS v1 requires a TH3F response");

    using VarType = correction::Variable::VarType;
    std::array<size_t, 5> indices{
        context.input_index("pt", VarType::real),
        context.input_index("eta", VarType::real),
        context.input_index("phi", VarType::real),
        context.input_index("m", VarType::real),
        context.input_index("detector_eta", VarType::real),
    };
    return std::make_shared<LargeJmsNode>(indices, std::move(response), settings);
  });
}
