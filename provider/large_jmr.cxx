#include "large_jmr.h"

#include "histogram.h"

#include <JetUncertainties/JetHelpers.h>
#include <TAxis.h>
#include <TH1.h>
#include <TLorentzVector.h>
#include <TMath.h>
#include <correction_plugin.h>
#include <rapidjson/document.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace jets {

thread_local const TruthContext* TruthContext::active_ = nullptr;

TruthContext::TruthContext(const P4* jets, std::size_t jet_count,
                           const std::size_t* offsets, std::size_t offset_count)
    : jets_(jets), offsets_(offsets), event_count_(offset_count > 0 ? offset_count - 1 : 0) {
  if (active_) throw std::logic_error("Nested truth-jet contexts are unsupported");
  if (!offsets || offset_count == 0 || (jet_count != 0 && !jets) ||
      offsets[0] != 0 || offsets[event_count_] != jet_count ||
      !std::is_sorted(offsets, offsets + offset_count)) {
    throw std::invalid_argument("Invalid truth-jet buffer and event offsets");
  }
  active_ = this;
}

TruthContext::~TruthContext() { active_ = nullptr; }

TruthJetRange TruthContext::event(std::int64_t event_index) {
  if (!active_) throw std::logic_error("No truth-jet context is active");
  if (event_index < 0 || static_cast<std::uint64_t>(event_index) >= active_->event_count_) {
    throw std::out_of_range("Truth-jet event index is outside the bound chunk");
  }
  const auto index = static_cast<std::size_t>(event_index);
  const auto first = active_->offsets_[index];
  const auto last = active_->offsets_[index + 1];
  if (first == last) return {nullptr, 0};
  return {active_->jets_ + first, last - first};
}

const P4* nearest_truth_jet(const P4& reco, TruthJetRange truth, double maximum_delta_r) {
  const TLorentzVector reco_cartesian(reco.Px(), reco.Py(), reco.Pz(), reco.E());
  const P4* match = nullptr;
  double closest_distance = maximum_delta_r;
  for (std::size_t index = 0; index < truth.size; ++index) {
    const auto& candidate = truth.data[index];
    const TLorentzVector candidate_cartesian(candidate.Px(), candidate.Py(), candidate.Pz(), candidate.E());
    const float distance = static_cast<float>(reco_cartesian.DeltaR(candidate_cartesian));
    // Strict comparison after the float cast keeps the first candidate on ties.
    if (distance < closest_distance) {
      closest_distance = distance;
      match = &candidate;
    }
  }
  return match;
}

}  // namespace jets

namespace {

double interpolate_in_range(const TH1& histogram, double x, double y) {
  const int xbin = histogram.GetXaxis()->FindFixBin(x);
  const int ybin = histogram.GetYaxis()->FindFixBin(y);
  if (xbin < 1 || xbin > histogram.GetNbinsX() || ybin < 1 || ybin > histogram.GetNbinsY()) {
    return 0.0;
  }
  return JetHelpers::Interpolate(&histogram, x, y);
}

double clamp_inside(double value, const TAxis& axis, double offset) {
  const double low = axis.GetBinLowEdge(1);
  const double high = axis.GetBinLowEdge(axis.GetNbins() + 1);
  // The original compares each limit against the original coordinate.
  if (value >= high) return high - offset;
  if (value <= low) return low + offset;
  return value;
}

double jmr_uncertainty(const jets::P4& reco, const jets::LargeJmrVariation& variation,
                       const jets::LargeJmrSettings& settings) {
  const TH1& histogram = variation.histogram;
  if (variation.binning == jets::JmrBinning::pt_abs_mass) {
    const float pt_gev = static_cast<float>(reco.Pt() * settings.mev_to_gev);
    const float mass_gev = static_cast<float>(reco.M() * settings.mev_to_gev);
    return interpolate_in_range(histogram, pt_gev, mass_gev) * variation.parameter;
  }

  const double energy = reco.E();
  const double log_mass_over_energy = TMath::Log(reco.M() / energy);
  if (std::isnan(log_mass_over_energy)) return 0.0;
  const double x = clamp_inside(energy * settings.mev_to_gev, *histogram.GetXaxis(), settings.axis_edge_offset);
  const double y = clamp_inside(log_mass_over_energy, *histogram.GetYaxis(), settings.axis_edge_offset);
  const double z = clamp_inside(std::abs(reco.Eta()), *histogram.GetZaxis(), settings.axis_edge_offset);
  return JetHelpers::Interpolate(&histogram, x, y, z) * variation.parameter;
}

}  // namespace

namespace jets {

LargeJmrResult apply_large_jmr(const P4& reco, std::optional<int> truth_label,
                               TruthJetRange truth, const TH1& response,
                               const LargeJmrVariation& variation,
                               const LargeJmrSettings& settings) {
  if (std::abs(reco.Eta()) > settings.max_abs_eta || reco.M() > settings.max_mass_mev ||
      reco.Pt() > settings.max_pt_mev || reco.M() <= 0.0) {
    return {reco, JmrStatus::out_of_validity, false};
  }
  const P4* truth_match = nearest_truth_jet(reco, truth, settings.max_match_delta_r);
  if (!truth_match) return {reco, JmrStatus::out_of_validity, false};
  if (!truth_label) return {reco, JmrStatus::error, false};

  const auto index = static_cast<std::size_t>(*truth_label);
  const std::string& topology =
      index < settings.topology_by_label.size() ? settings.topology_by_label[index] : settings.fallback_topology;
  if (topology == "no_match") return {reco, JmrStatus::out_of_validity, false};

  // xAOD copies the matched jet through the same float stores before reading m.
  const double truth_mass = stored_p4(*truth_match).M();
  double average_response = interpolate_in_range(
      response, reco.Pt() * settings.mev_to_gev, truth_mass * settings.mev_to_gev);
  if (average_response == 0.0) average_response = settings.zero_response_fallback;

  const double error =
      variation.topology_filter == "All" || variation.topology_filter.find(topology) != std::string_view::npos
          ? jmr_uncertainty(reco, variation, settings)
          : 0.0;
  const double scale = settings.nominal_scale;
  const double resolution = settings.nominal_resolution + error;
  if (!(std::abs(scale - 1.0) > settings.minimum_shift ||
        std::abs(resolution - 1.0) > settings.minimum_shift)) {
    return {reco, JmrStatus::ok, false};
  }

  const double mass = reco.M();
  const double smeared_mass = mass * scale + (mass - average_response * truth_mass) * (resolution - scale);
  const P4 varied(reco.Pt(), reco.Eta(), reco.Phi(), smeared_mass);
  return {varied, JmrStatus::ok, true};
}

}  // namespace jets

namespace {

using Json = rapidjson::Value;
using VarType = correction::Variable::VarType;
constexpr std::size_t jmr_output_count = 6;

const Json& field(const Json& object, const char* key) {
  if (!object.IsObject() || !object.HasMember(key)) {
    throw std::invalid_argument(std::string("Missing Large-R JMR field: ") + key);
  }
  return object[key];
}

std::string string_field(const Json& object, const char* key) {
  const auto& value = field(object, key);
  if (!value.IsString()) throw std::invalid_argument(std::string("Expected Large-R JMR string: ") + key);
  return {value.GetString(), value.GetStringLength()};
}

double positive_number(const correction::PluginConfig& config, const char* key) {
  double value = config.real(key);
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument(std::string("Expected positive Large-R JMR number: ") + key);
  }
  return value;
}

float positive_float(const correction::PluginConfig& config, const char* key) {
  double value = positive_number(config, key);
  if (value > std::numeric_limits<float>::max() || static_cast<float>(value) == 0.0f) {
    throw std::invalid_argument(std::string("Large-R JMR number is outside float range: ") + key);
  }
  return static_cast<float>(value);
}

jets::JmrBinning jmr_binning(const std::string& name) {
  if (name == "PtAbsMass") return jets::JmrBinning::pt_abs_mass;
  if (name == "eLOGmOeAbsEta") return jets::JmrBinning::energy_log_mass_abs_eta;
  throw std::invalid_argument("Unsupported Large-R JMR binning: " + name);
}

struct SourceComponent {
  jets::JmrBinning binning;
  std::string topology;
  std::unique_ptr<const TH1> histogram;
};

struct State {
  std::size_t component_index;
  float parameter;
};

struct InputIndices {
  std::size_t pt;
  std::size_t eta;
  std::size_t phi;
  std::size_t mass;
  std::size_t truth_label;
  std::size_t truth_label_available;
  std::size_t systematic;
  std::size_t event_index;
};

class LargeJmrNode final : public correction::PluginNode {
 public:
  LargeJmrNode(InputIndices indices, jets::LargeJmrSettings settings,
               std::unique_ptr<const TH1> response, std::vector<SourceComponent> components,
               std::unordered_map<std::string, State> states)
      : indices_(indices), settings_(std::move(settings)), response_(std::move(response)),
        components_(std::move(components)), states_(std::move(states)) {}

  std::size_t output_size() const override { return jmr_output_count; }

  double evaluate(const std::vector<correction::Variable::Type>&) const override {
    throw std::logic_error("Large-R JMR has an array output");
  }

  void evaluate_array(const std::vector<correction::Variable::Type>& values,
                      double* output, std::size_t size) const override {
    if (!output || size < output_size()) throw std::invalid_argument("Large-R JMR output is too small");
    const auto pt = std::get<double>(values.at(indices_.pt));
    const auto eta = std::get<double>(values.at(indices_.eta));
    const auto phi = std::get<double>(values.at(indices_.phi));
    const auto mass = std::get<double>(values.at(indices_.mass));
    const auto& systematic = std::get<std::string>(values.at(indices_.systematic));
    const auto found = states_.find(systematic);
    if (found == states_.end()) {
      output[0] = pt;
      output[1] = eta;
      output[2] = phi;
      output[3] = mass;
      output[4] = std::numeric_limits<double>::quiet_NaN();
      output[5] = 0.0;
      return;
    }

    const auto available = std::get<std::int64_t>(values.at(indices_.truth_label_available));
    const auto input_label = std::get<std::int64_t>(values.at(indices_.truth_label));
    if (available < 0 || available > 1 ||
        (available && (input_label < std::numeric_limits<int>::min() ||
                       input_label > std::numeric_limits<int>::max()))) {
      throw std::invalid_argument("Invalid Large-R JMR truth label or availability");
    }
    const auto truth = jets::TruthContext::event(std::get<std::int64_t>(values.at(indices_.event_index)));
    const auto& state = found->second;
    const auto& component = components_.at(state.component_index);
    const jets::LargeJmrVariation variation{component.binning, component.topology,
                                             *component.histogram, state.parameter};
    const auto result = jets::apply_large_jmr(jets::P4(pt, eta, phi, mass),
        available ? std::optional<int>(static_cast<int>(input_label)) : std::nullopt,
        truth, *response_, variation, settings_);
    output[0] = result.write_p4 ? static_cast<float>(result.p4.Pt()) : pt;
    output[1] = result.write_p4 ? static_cast<float>(result.p4.Eta()) : eta;
    output[2] = result.write_p4 ? static_cast<float>(result.p4.Phi()) : phi;
    output[3] = result.write_p4 ? static_cast<float>(result.p4.M()) : mass;
    output[4] = static_cast<double>(result.status);
    output[5] = 1.0;
  }

 private:
  InputIndices indices_;
  jets::LargeJmrSettings settings_;
  std::unique_ptr<const TH1> response_;
  std::vector<SourceComponent> components_;
  std::unordered_map<std::string, State> states_;
};

}  // namespace

void register_large_jmr(correction::PluginRegistry& registry) {
  registry.register_node("org.atlas.jets.large_jmr", 1,
      [](const correction::PluginConfig& config, const correction::PluginContext& context) {
    config.require_keys({"response", "components", "max_abs_eta", "max_mass_mev", "max_pt_mev",
                         "mev_to_gev", "max_match_delta_r", "axis_edge_offset", "minimum_shift",
                         "nominal_scale", "nominal_resolution", "zero_response_fallback",
                         "fallback_topology", "topology_by_label"});
    if (context.output_size() != jmr_output_count || context.child_count() != 0) {
      throw std::invalid_argument("Large-R JMR requires six output values and no children");
    }

    rapidjson::Document document;
    document.Parse<rapidjson::kParseFullPrecisionFlag>(config.json().c_str());
    if (document.HasParseError() || !document.IsObject()) {
      throw std::invalid_argument("Invalid Large-R JMR configuration");
    }

    jets::LargeJmrSettings settings{
        positive_float(config, "max_abs_eta"),
        positive_float(config, "max_mass_mev"),
        positive_float(config, "max_pt_mev"),
        positive_float(config, "mev_to_gev"),
        positive_number(config, "max_match_delta_r"),
        positive_number(config, "axis_edge_offset"),
        positive_number(config, "minimum_shift"),
        positive_number(config, "nominal_scale"),
        positive_number(config, "nominal_resolution"),
        positive_number(config, "zero_response_fallback"),
        config.string("fallback_topology"),
        {}};

    const auto& labels = field(document, "topology_by_label");
    if (!labels.IsArray() || labels.Empty() || settings.fallback_topology.empty()) {
      throw std::invalid_argument("Large-R JMR requires a truth-label map and fallback topology");
    }
    for (const auto& topology : labels.GetArray()) {
      if (!topology.IsString() || topology.GetStringLength() == 0) {
        throw std::invalid_argument("Invalid Large-R JMR truth topology");
      }
      settings.topology_by_label.emplace_back(topology.GetString(), topology.GetStringLength());
    }

    auto response = jets::restore_histogram(field(document, "response"));
    if (response->GetDimension() != 2) {
      throw std::invalid_argument("Large-R JMR response needs a TH2");
    }

    const auto& source_components = field(document, "components");
    if (!source_components.IsArray() || source_components.Empty()) {
      throw std::invalid_argument("Large-R JMR component list is empty");
    }
    std::vector<SourceComponent> components;
    std::unordered_map<std::string, State> states;
    for (const auto& item : source_components.GetArray()) {
      const auto binning = jmr_binning(string_field(item, "binning"));
      const auto topology = string_field(item, "topology");
      auto histogram = jets::restore_histogram(field(item, "histogram"));
      const bool expected_histogram = binning == jets::JmrBinning::pt_abs_mass
          ? histogram->GetDimension() == 2
          : std::string_view(histogram->ClassName()) == "TH3F";
      if (topology.empty() || !expected_histogram) {
        throw std::invalid_argument("Invalid Large-R JMR component histogram or topology");
      }

      const auto& variations = field(item, "states");
      if (!variations.IsObject() || variations.MemberCount() != 2) {
        throw std::invalid_argument("Large-R JMR components require up and down states");
      }
      double sum_parameters = 0.0;
      for (const auto& state : variations.GetObject()) {
        if (!state.value.IsNumber() || state.name.GetStringLength() == 0) {
          throw std::invalid_argument("Invalid Large-R JMR state name or parameter");
        }
        const double parameter = state.value.GetDouble();
        if (std::abs(parameter) != 1.0) {
          throw std::invalid_argument("Large-R JMR v1 uses only +1 and -1");
        }
        const std::string name(state.name.GetString(), state.name.GetStringLength());
        if (!states.emplace(name, State{components.size(), static_cast<float>(parameter)}).second) {
          throw std::invalid_argument("Duplicate Large-R JMR state: " + name);
        }
        sum_parameters += parameter;
      }
      if (sum_parameters != 0.0) {
        throw std::invalid_argument("Large-R JMR component is missing a variation sign");
      }
      components.push_back({binning, topology, std::move(histogram)});
    }

    InputIndices indices{
        context.input_index("pt", VarType::real),
        context.input_index("eta", VarType::real),
        context.input_index("phi", VarType::real),
        context.input_index("m", VarType::real),
        context.input_index("truth_label", VarType::integer),
        context.input_index("truth_label_available", VarType::integer),
        context.input_index("systematic", VarType::string),
        context.input_index("event_index", VarType::integer),
    };
    return std::make_shared<LargeJmrNode>(indices, std::move(settings), std::move(response),
                                          std::move(components), std::move(states));
  });
}
