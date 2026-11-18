#include "small_gsc.h"

#include "jet_histogram.h"
#include "jet_common.h"

#include <correction_plugin.h>
#include <rapidjson/document.h>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using correction::PluginConfig;
using correction::PluginContext;
using correction::Variable;
using Values = std::vector<Variable::Type>;

enum Family : size_t { charged, tile0, em3, n_track, width, n_families };
using Tables = std::array<std::vector<std::unique_ptr<const jets::CalibrationHistogram>>, n_families>;

double axis_coordinate(const jets::CalibrationAxis& axis, double coordinate, double offset) {
  const int bin = axis.FindBin(coordinate);
  if (bin < axis.GetFirst()) return axis.GetBinLowEdge(axis.GetFirst()) + offset;
  if (bin > axis.GetLast()) return axis.GetBinUpEdge(axis.GetLast()) - offset;
  return coordinate;
}

struct Recipe {
  Tables response;
  double eta_width;
  double mev_per_gev;
  double edge_offset;

  double interpolate(const jets::CalibrationHistogram& table, double pt, double moment) const {
    return table.Interpolate(axis_coordinate(*table.GetXaxis(), pt, edge_offset),
                             axis_coordinate(*table.GetYaxis(), moment, edge_offset));
  }

  double scale(double pt, double abs_eta, double charged_fraction, float tile_fraction,
               float em_fraction, int n_tracks, float track_width) const {
    const double coordinate = abs_eta / eta_width;
    if (!std::isfinite(coordinate) || coordinate > std::numeric_limits<int>::max())
      throw std::domain_error("small-R GSC detector eta cannot be mapped to its integer bin");
    const auto eta_bin = static_cast<size_t>(static_cast<int>(coordinate));
    const double pt_gev = pt / mev_per_gev;
    double factor = 1.;
    const auto apply = [&](Family family, double lookup_pt, double moment, bool skip) {
      if (!skip && eta_bin < response[family].size())
        factor *= 1. / interpolate(*response[family][eta_bin], lookup_pt, moment);
    };
    apply(charged, pt_gev, charged_fraction, charged_fraction <= 0);
    apply(tile0, pt_gev * factor, tile_fraction, tile_fraction < 0);
    apply(em3, pt_gev * factor, em_fraction, em_fraction <= 0);
    apply(n_track, pt_gev * factor, n_tracks, n_tracks <= 0);
    apply(width, pt_gev * factor, track_width, track_width <= 0);
    return factor;
  }
};

Recipe read_recipe(const PluginConfig& config) {
  Recipe recipe{{}, config.real("abs_eta_width"), config.real("mev_per_gev"),
                config.real("axis_edge_offset")};
  if (!std::isfinite(recipe.eta_width) || recipe.eta_width <= 0 ||
      !std::isfinite(recipe.mev_per_gev) || recipe.mev_per_gev <= 0 ||
      !std::isfinite(recipe.edge_offset) || recipe.edge_offset <= 0)
    throw std::invalid_argument("invalid small-R GSC numerical settings");

  rapidjson::Document json;
  json.Parse<rapidjson::kParseFullPrecisionFlag>(config.json().c_str());
  if (json.HasParseError() || !json.IsObject() || !json["responses"].IsObject())
    throw std::invalid_argument("invalid small-R GSC response configuration");
  const auto& responses = json["responses"];
  constexpr std::array<const char*, n_families> names{"chargedFraction", "Tile0", "EM3", "nTrk", "trackWIDTH"};
  if (responses.MemberCount() != names.size())
    throw std::invalid_argument("small-R GSC requires five families of responses");
  for (size_t i = 0; i < names.size(); ++i) {
    if (!responses.HasMember(names[i]) || !responses[names[i]].IsArray() ||
        responses[names[i]].Size() < 3)
      throw std::invalid_argument("missing small-R GSC response family");
    for (const auto& record : responses[names[i]].GetArray()) {
      auto table = jets::restore_calibration_histogram(record);
      if (std::string_view(table->ClassName()) != "TH2F")
        throw std::invalid_argument("small-R GSC requires a TH2F response");
      recipe.response[i].emplace_back(std::move(table));
    }
  }
  return recipe;
}

std::array<size_t, 4> vector_fields(const PluginContext& context, const char* pt,
                                   const char* eta, const char* phi, const char* mass) {
  return {context.input_index(pt, Variable::VarType::real),
          context.input_index(eta, Variable::VarType::real),
          context.input_index(phi, Variable::VarType::real),
          context.input_index(mass, Variable::VarType::real)};
}

class GscNode final : public correction::PluginNode {
public:
  GscNode(const PluginContext& context, Recipe recipe)
      : eta_jes_(vector_fields(context, "pt", "eta", "phi", "m")),
        constit_(vector_fields(context, "constituent_pt", "constituent_eta", "constituent_phi", "constituent_m")),
        detector_eta_(context.input_index("detector_eta", Variable::VarType::real)),
        sum_charged_(context.input_index("sum_pt_charged_pfo", Variable::VarType::real)),
        n_trk_(context.input_index("n_trk", Variable::VarType::integer)),
        track_width_(context.input_index("track_width", Variable::VarType::real)),
        em_b3_(context.input_index("em_b3", Variable::VarType::real)),
        em_e3_(context.input_index("em_e3", Variable::VarType::real)),
        tile_bar0_(context.input_index("tile_bar0", Variable::VarType::real)),
        tile_ext0_(context.input_index("tile_ext0", Variable::VarType::real)),
        recipe_(std::move(recipe)) {}

  size_t output_size() const override { return 4; }

  double evaluate(const Values&) const override {
    throw std::logic_error("small-R GSC returns four coordinates");
  }

  void evaluate_array(const Values& values, double* output, size_t size) const override {
    if (!output || size < output_size())
      throw std::invalid_argument("small-R GSC output needs four coordinates");
    const auto aux = [&values](size_t i) { return static_cast<float>(std::get<double>(values.at(i))); };
    const auto read_vector = [&aux](const std::array<size_t, 4>& index) {
      return jets::P4(aux(index[0]), aux(index[1]), aux(index[2]), aux(index[3]));
    };
    const auto tracks = std::get<int64_t>(values.at(n_trk_));
    if (tracks < std::numeric_limits<int>::min() || tracks > std::numeric_limits<int>::max())
      throw std::domain_error("small-R GSC track count exceeds the xAOD integer range");

    const auto constituent = read_vector(constit_);
    const auto start = jets::stored_p4(read_vector(eta_jes_));
    const double charged_fraction = aux(sum_charged_) / constituent.Pt();
    const float constituent_energy = static_cast<float>(constituent.E());
    const float em_fraction = (aux(em_b3_) + aux(em_e3_)) / constituent_energy;
    const float tile_fraction = (aux(tile_bar0_) + aux(tile_ext0_)) / constituent_energy;
    const double abs_eta = std::abs(static_cast<double>(aux(detector_eta_)));
    const double factor = recipe_.scale(start.Pt(), abs_eta, charged_fraction, tile_fraction,
                                        em_fraction, static_cast<int>(tracks), aux(track_width_));
    const auto calibrated = start * factor;
    output[0] = static_cast<float>(calibrated.Pt());
    output[1] = static_cast<float>(calibrated.Eta());
    output[2] = static_cast<float>(calibrated.Phi());
    output[3] = static_cast<float>(calibrated.M());
  }

private:
  const std::array<size_t, 4> eta_jes_, constit_;
  const size_t detector_eta_, sum_charged_, n_trk_, track_width_, em_b3_, em_e3_, tile_bar0_, tile_ext0_;
  const Recipe recipe_;
};
}  // namespace

void register_small_gsc(correction::PluginRegistry& registry) {
  registry.register_node("org.atlas.jets.small_gsc", 1,
      [](const PluginConfig& config, const PluginContext& context) {
        config.require_keys({"abs_eta_width", "mev_per_gev", "axis_edge_offset", "responses"});
        if (context.output_size() != 4 || context.child_count() != 0)
          throw std::invalid_argument("small-R GSC requires four outputs and no children");
        return std::make_shared<GscNode>(context, read_recipe(config));
      });
}
