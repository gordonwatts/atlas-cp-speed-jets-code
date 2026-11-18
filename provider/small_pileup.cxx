#include "small_pileup.h"

#include "jet_common.h"

#include <correction_plugin.h>
#include <rapidjson/document.h>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using correction::PluginConfig;
using correction::PluginContext;
using correction::Variable;
using Values = std::vector<Variable::Type>;

std::vector<double> numbers(const rapidjson::Value& input) {
  if (!input.IsArray()) throw std::invalid_argument("small-R residual needs numeric arrays");
  std::vector<double> result;
  result.reserve(input.Size());
  for (const auto& item : input.GetArray()) {
    if (!item.IsNumber() || !std::isfinite(item.GetDouble()))
      throw std::invalid_argument("small-R residual coefficients and edges must be finite numbers");
    result.push_back(item.GetDouble());
  }
  return result;
}

struct Residual {
  std::vector<double> edges;
  std::vector<double> mu;
  std::vector<double> npv;
  double mu_ref;
  double npv_ref;
  double mu_factor;
  double mev_per_gev;
  double reset_pt_gev;

  double offset(double abs_eta, double average_mu, double n_vertices) const {
    double alpha = mu.front();
    double beta = npv.front();
    for (size_t bin = 1; bin < edges.size(); ++bin) {
      const double low = edges[bin - 1];
      const double width = edges[bin] - low;
      if (abs_eta < low + width) {
        alpha += mu[bin] * (abs_eta - low);
        beta += npv[bin] * (abs_eta - low);
        const double corrected_mu = average_mu * mu_factor;
        return (alpha * (corrected_mu - mu_ref) + beta * (n_vertices - npv_ref)) * mev_per_gev;
      }
      alpha += width * mu[bin];
      beta += width * npv[bin];
    }
    // In this region ResidualOffsetCorrection reads an out-of-bounds coefficient.
    throw std::domain_error("small-R residual |eta| is outside its coefficient domain");
  }
};

Residual read_residual(const PluginConfig& config) {
  rapidjson::Document json;
  json.Parse<rapidjson::kParseFullPrecisionFlag>(config.json().c_str());
  if (json.HasParseError() || !json.IsObject())
    throw std::invalid_argument("invalid small-R residual configuration");
  Residual recipe{
      numbers(json["abs_eta_edges"]), numbers(json["mu_coefficients"]), numbers(json["npv_coefficients"]),
      config.real("mu_ref"), config.real("npv_ref"), config.real("mu_factor"),
      config.real("mev_per_gev"), config.real("negative_reset_pt_gev")};
  if (recipe.edges.size() < 3 || recipe.edges.size() != recipe.mu.size() ||
      recipe.edges.size() != recipe.npv.size() || recipe.edges.front() != 0.0 ||
      !std::isfinite(recipe.mu_ref) || !std::isfinite(recipe.npv_ref) ||
      !std::isfinite(recipe.mu_factor) || !std::isfinite(recipe.mev_per_gev) ||
      !std::isfinite(recipe.reset_pt_gev) || recipe.mev_per_gev <= 0 || recipe.reset_pt_gev <= 0)
    throw std::invalid_argument("invalid small-R residual settings");
  for (size_t i = 1; i < recipe.edges.size(); ++i)
    if (recipe.edges[i] <= recipe.edges[i - 1])
      throw std::invalid_argument("small-R residual eta edges must be increasing");
  return recipe;
}

class PileupNode final : public correction::PluginNode {
public:
  PileupNode(const PluginContext& context, Residual recipe)
      : constit_{context.input_index("pt", Variable::VarType::real),
                 context.input_index("eta", Variable::VarType::real),
                 context.input_index("phi", Variable::VarType::real),
                 context.input_index("m", Variable::VarType::real)},
        area_(context.input_index("area_pt", Variable::VarType::real)),
        rho_(context.input_index("rho", Variable::VarType::real)),
        mu_(context.input_index("mu", Variable::VarType::real)),
        npv_(context.input_index("npv", Variable::VarType::integer)),
        recipe_(std::move(recipe)) {}

  size_t output_size() const override { return 4; }

  double evaluate(const Values&) const override {
    throw std::logic_error("small-R pile-up returns four coordinates");
  }

  void evaluate_array(const Values& values, double* output, size_t size) const override {
    if (!output || size < output_size())
      throw std::invalid_argument("small-R pile-up output needs four coordinates");
    const auto aux = [&values](size_t i) { return static_cast<float>(std::get<double>(values.at(i))); };
    const auto npv = std::get<int64_t>(values.at(npv_));
    if (npv < 0 || npv > std::numeric_limits<int>::max())
      throw std::domain_error("small-R nominal NPV must be a nonnegative vertex count");

    const jets::P4 input(aux(constit_[0]), aux(constit_[1]), aux(constit_[2]), aux(constit_[3]));
    // setStartP4 writes the constituent-scale vector to the assigned-scale
    // decorations, and the tool immediately re-reads those stored floats.
    const auto start = jets::stored_p4(input);
    if (start.E() < start.M())
      throw std::domain_error("small-R pile-up constituent energy is less than its mass");

    const double offset = recipe_.offset(std::abs(start.Eta()), aux(mu_), static_cast<double>(npv));
    const double rho = std::get<double>(values.at(rho_));
    const double pt = start.Pt();
    const double shifted = pt - rho * aux(area_) - offset;
    const double scale = shifted >= 0 ? shifted / pt : recipe_.reset_pt_gev * recipe_.mev_per_gev / pt;
    const auto calibrated = start * scale;

    // JetPileupScaleMomentum and the assigned scale each write these four
    // float aux values. Return their promotion before any later re-read.
    output[0] = static_cast<float>(calibrated.Pt());
    output[1] = static_cast<float>(calibrated.Eta());
    output[2] = static_cast<float>(calibrated.Phi());
    output[3] = static_cast<float>(calibrated.M());
  }

private:
  const std::array<size_t, 4> constit_;
  const size_t area_, rho_, mu_, npv_;
  const Residual recipe_;
};
}  // namespace

void register_small_pileup(correction::PluginRegistry& registry) {
  registry.register_node("org.atlas.jets.small_pileup", 1,
      [](const PluginConfig& config, const PluginContext& context) {
        config.require_keys({"abs_eta_edges", "mu_coefficients", "npv_coefficients",
                             "mu_ref", "npv_ref", "mu_factor", "mev_per_gev", "negative_reset_pt_gev"});
        if (context.output_size() != 4 || context.child_count() != 0)
          throw std::invalid_argument("small-R pile-up requires four outputs and no children");
        return std::make_shared<PileupNode>(context, read_residual(config));
      });
}
