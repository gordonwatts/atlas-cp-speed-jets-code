#include "small_jvt.h"

#include "../portable/lwtnn.h"

#include <correction_plugin.h>
#include <rapidjson/document.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using correction::PluginConfig;
using correction::PluginContext;
using correction::PluginNode;
using correction::Variable;
using Values = std::vector<Variable::Type>;
using Json = rapidjson::Value;
using Network = jets::portable::JvtNetwork;

enum Status : int { error = -1, skipped = 0, ok = 1, out_of_validity = 2 };

float as_float(const Values& values, size_t index) {
  return static_cast<float>(std::get<double>(values.at(index)));
}

int64_t as_code(const Values& values, size_t index, int64_t low, int64_t high) {
  const auto value = std::get<int64_t>(values.at(index));
  if (value < low || value > high)
    throw std::invalid_argument("invalid NNJvt input code");
  return value;
}

size_t real_input(const PluginContext& context, const char* name) {
  return context.input_index(name, Variable::VarType::real);
}

size_t int_input(const PluginContext& context, const char* name) {
  return context.input_index(name, Variable::VarType::integer);
}

const Json& field(const Json& object, const char* name) {
  if (!object.IsObject() || !object.HasMember(name))
    throw std::invalid_argument(std::string("missing NNJvt cut field: ") + name);
  return object[name];
}

float finite_float(const Json& value) {
  if (!value.IsNumber())
    throw std::invalid_argument("expected a number in NNJvt cuts");
  const float converted = static_cast<float>(value.GetDouble());
  if (!std::isfinite(converted))
    throw std::invalid_argument("non-finite value in NNJvt cuts");
  return converted;
}

std::vector<float> floats(const Json& values) {
  if (!values.IsArray())
    throw std::invalid_argument("NNJvt cuts require numeric arrays");
  std::vector<float> result;
  result.reserve(values.Size());
  for (const auto& item : values.GetArray()) result.push_back(finite_float(item));
  return result;
}

struct CutMap {
  std::vector<float> pt_edges;
  std::vector<float> eta_edges;
  std::vector<std::vector<float>> cuts;

  bool bins(float pt, float eta, size_t& pt_bin, size_t& eta_bin) const {
    // lower_bound is the exact release convention. Pt underflow is evaluated
    // in the first bin, and eta underflow is outside the model.
    const auto pt_edge = std::lower_bound(pt_edges.begin(), pt_edges.end(), pt);
    const auto eta_edge = std::lower_bound(eta_edges.begin(), eta_edges.end(), eta);
    if (pt_edge == pt_edges.end() || eta_edge == eta_edges.begin() || eta_edge == eta_edges.end())
      return false;
    pt_bin = pt_edge == pt_edges.begin() ? 0 : static_cast<size_t>(pt_edge - pt_edges.begin() - 1);
    eta_bin = static_cast<size_t>(eta_edge - eta_edges.begin() - 1);
    return true;
  }

  float cut(float pt, float eta) const {
    size_t pt_bin, eta_bin;
    return bins(pt, eta, pt_bin, eta_bin) ? cuts[pt_bin][eta_bin] : -1.f;
  }
};

CutMap read_cuts(const PluginConfig& config) {
  rapidjson::Document json;
  json.Parse<rapidjson::kParseFullPrecisionFlag>(config.json().c_str());
  if (json.HasParseError()) throw std::invalid_argument("malformed NNJvt cut configuration");
  const auto& data = field(json, "cut_map");
  const float mev_per_gev = static_cast<float>(config.real("mev_per_gev"));
  if (mev_per_gev != 1000.f)
    throw std::invalid_argument("NNJvt cut conversion expects 1000 MeV per GeV");

  CutMap map{floats(field(data, "pt_edges_gev")), floats(field(data, "eta_edges")), {}};
  for (float& edge : map.pt_edges) edge *= mev_per_gev;
  for (const auto* edges : {&map.pt_edges, &map.eta_edges}) {
    if (edges->size() < 2 || !std::isfinite(edges->back()) ||
        std::adjacent_find(edges->begin(), edges->end(), std::greater_equal<float>()) != edges->end())
      throw std::invalid_argument("invalid NNJvt cut-map edges");
  }
  const auto& cuts = field(data, "cuts");
  if (!cuts.IsArray() || cuts.Size() + 1 != map.pt_edges.size())
    throw std::invalid_argument("wrong NNJvt cut-map pT dimension");
  for (const auto& row : cuts.GetArray()) {
    auto entries = floats(row);
    if (entries.size() + 1 != map.eta_edges.size())
      throw std::invalid_argument("wrong NNJvt cut-map eta dimension");
    map.cuts.push_back(std::move(entries));
  }
  return map;
}

std::unique_ptr<const Network> read_network(const PluginConfig& config) {
  std::istringstream input(config.string("network_json"));
  const lwt::GraphConfig graph = lwt::parse_json_graph(input);
  if (graph.inputs.size() != 1 || !graph.input_sequences.empty() || graph.outputs.size() != 1)
    throw std::invalid_argument("NNJvt expects one scalar input node and one output node");
  lwt::InputOrder order;
  order.scalar.emplace_back(graph.inputs.front().name,
                            std::vector<std::string>{"Rpt", "JVFCorr", "ptbin", "etabin"});
  return std::make_unique<const Network>(graph, order);
}

struct Range {
  float min_pt;
  float max_pt;
  float min_eta;
  float max_eta;

  bool includes(float pt, float eta) const {
    if (pt < min_pt || pt > max_pt) return false;
    const float abs_eta = std::abs(eta);
    return abs_eta >= min_eta && abs_eta <= max_eta;
  }
};

Range read_range(const PluginConfig& config) {
  const Range result{static_cast<float>(config.real("min_pt")),
                     static_cast<float>(config.real("max_pt")),
                     static_cast<float>(config.real("min_abs_eta")),
                     static_cast<float>(config.real("max_abs_eta"))};
  if (!std::isfinite(result.min_pt) || !std::isfinite(result.max_pt) ||
      !std::isfinite(result.min_eta) || !std::isfinite(result.max_eta) ||
      result.min_pt > result.max_pt || result.min_eta > result.max_eta)
    throw std::invalid_argument("invalid NNJvt validity range");
  return result;
}

class ArrayNode : public PluginNode {
public:
  double evaluate(const Values&) const override {
    throw std::logic_error("use array evaluation for NNJvt corrections");
  }

protected:
  void check_buffer(double* output, size_t size) const {
    if (!output || size < output_size())
      throw std::invalid_argument("NNJvt correction output buffer too small");
  }
};

class NnJvtUpdate final : public ArrayNode {
public:
  NnJvtUpdate(const PluginConfig& config, const PluginContext& context)
      : pt_(real_input(context, "pt")), eta_(real_input(context, "eta")),
        sum_(real_input(context, "sum_pt_trk")), jvf_(real_input(context, "jvf_corr")),
        vertex_(int_input(context, "vertex_kind")), features_(int_input(context, "feature_status")),
        max_pt_(static_cast<float>(config.real("max_pt"))),
        cuts_(read_cuts(config)), network_(read_network(config)) {
    if (!std::isfinite(max_pt_)) throw std::invalid_argument("invalid NNJvt maximum pT");
  }

  size_t output_size() const override { return 4; }

  void evaluate_array(const Values& values, double* output, size_t size) const override {
    check_buffer(output, size);
    const auto vertex = as_code(values, vertex_, 0, 2);
    // Initialize to native decorator defaults. Status says whether any of them
    // were written and lets the adapter keep inherited decorations on a skip.
    output[0] = -1;
    output[1] = 0;
    output[2] = 1;
    output[3] = ok;
    if (vertex == 0) {
      output[3] = error;
      return;
    }
    if (vertex == 1) return;  // Single NoVtx sentinel.

    const auto features = as_code(values, features_, -1, 1);
    if (features != 1) {
      output[3] = features == 0 ? skipped : error;
      return;
    }

    const float pt = as_float(values, pt_);
    const float eta = as_float(values, eta_);
    const float track_sum = as_float(values, sum_);
    const float rpt = pt == 0.f ? 0.f : static_cast<float>(track_sum / static_cast<double>(pt));
    output[1] = rpt;
    size_t pt_bin, eta_bin;
    if (!(pt <= max_pt_) || !cuts_.bins(pt, eta, pt_bin, eta_bin)) return;

    jets::portable::JvtNetworkFeatures inputs(4);
    inputs << rpt, as_float(values, jvf_), static_cast<float>(pt_bin), static_cast<float>(eta_bin);
    const auto predicted = network_->compute(jets::portable::JvtNetworkNodes{std::move(inputs)});
    const float score = static_cast<float>(predicted(0));
    output[0] = score;
    output[2] = score > cuts_.cuts[pt_bin][eta_bin];
  }

private:
  const size_t pt_, eta_, sum_, jvf_, vertex_, features_;
  const float max_pt_;
  const CutMap cuts_;
  const std::unique_ptr<const Network> network_;
};

class NnJvtSelect final : public ArrayNode {
public:
  NnJvtSelect(const PluginConfig& config, const PluginContext& context)
      : pt_(real_input(context, "pt")), eta_(real_input(context, "eta")),
        score_(real_input(context, "score")), present_(int_input(context, "score_present")),
        range_(read_range(config)), cuts_(read_cuts(config)) {}

  size_t output_size() const override { return 2; }

  void evaluate_array(const Values& values, double* output, size_t size) const override {
    check_buffer(output, size);
    const float pt = as_float(values, pt_);
    const float eta = as_float(values, eta_);
    output[1] = ok;
    if (!range_.includes(pt, eta)) {
      output[0] = 1;
    } else if (!as_code(values, present_, 0, 1)) {
      output[0] = 0;
      output[1] = error;
    } else {
      output[0] = as_float(values, score_) > cuts_.cut(pt, eta);
    }
  }

private:
  const size_t pt_, eta_, score_, present_;
  const Range range_;
  const CutMap cuts_;
};

class NnJvtSf final : public ArrayNode {
public:
  NnJvtSf(const PluginConfig& config, const PluginContext& context)
      : pt_(real_input(context, "pt")), eta_(real_input(context, "eta")),
        pass_(int_input(context, "pass")), truth_(int_input(context, "is_jvt_hs")),
        variation_(int_input(context, "variation")), preselected_(int_input(context, "preselected")),
        range_(read_range(config)), valid_bin_(context.child(0)), sf_(context.child(1)),
        sf_error_(context.child(2)), efficiency_(context.child(3)), efficiency_error_(context.child(4)) {}

  size_t output_size() const override { return 3; }

  void evaluate_array(const Values& values, double* output, size_t size) const override {
    check_buffer(output, size);
    const auto sigma = as_code(values, variation_, -1, 1);
    if (!as_code(values, preselected_, 0, 1)) {
      write(output, -1.f, skipped);
      return;
    }
    const float pt = as_float(values, pt_);
    const float eta = as_float(values, eta_);
    if (!range_.includes(pt, eta)) {
      write(output, -1.f, out_of_validity);
      return;
    }
    const auto truth = as_code(values, truth_, -1, 1);
    if (truth != 1) {
      write(output, 1.f, truth == 0 ? ok : error);
      return;
    }

    // The underlying SF histograms have explicit variable axes. MultiBinning
    // implements their fixed-bin lookup once the xAOD inputs are float-rounded.
    const bool rounded = pt != std::get<double>(values.at(pt_)) ||
                         eta != std::get<double>(values.at(eta_));
    if (rounded) {
      auto rounded_values = values;
      rounded_values[pt_] = static_cast<double>(pt);
      rounded_values[eta_] = static_cast<double>(eta);
      evaluate_histograms(rounded_values, sigma, output);
    } else {
      evaluate_histograms(values, sigma, output);
    }
  }

private:
  static void write(double* output, float sf, Status status) {
    output[0] = sf;
    output[1] = status;
    output[2] = status == ok ? sf : 1.f;
  }

  void evaluate_histograms(const Values& values, int64_t sigma, double* output) const {
    if (valid_bin_->evaluate(values) == 0.) {
      write(output, -1.f, out_of_validity);
      return;
    }
    const float base_sf = static_cast<float>(sf_->evaluate(values));
    const float error_sf = static_cast<float>(sf_error_->evaluate(values));
    const float varied_sf = base_sf + static_cast<float>(sigma) * error_sf;
    if (as_code(values, pass_, 0, 1)) {
      write(output, varied_sf, ok);
      return;
    }
    const float base_eff = static_cast<float>(efficiency_->evaluate(values));
    const float error_eff = static_cast<float>(efficiency_error_->evaluate(values));
    const float varied_eff = base_eff + error_eff * static_cast<float>(sigma);
    write(output, (1 - varied_sf * varied_eff) / (1 - varied_eff), ok);
  }

  const size_t pt_, eta_, pass_, truth_, variation_, preselected_;
  const Range range_;
  const std::shared_ptr<const PluginNode> valid_bin_, sf_, sf_error_, efficiency_, efficiency_error_;
};

void check_children(const PluginContext& context, size_t outputs, size_t children) {
  if (context.output_size() != outputs || context.child_count() != children)
    throw std::invalid_argument("NNJvt correction shape or child count mismatch");
  for (size_t child = 0; child < children; ++child)
    if (context.child(child)->output_size() != 0)
      throw std::invalid_argument("NNJvt SF requires scalar histogram children");
}
}  // namespace

void register_small_jvt(correction::PluginRegistry& registry) {
  registry.register_node("org.atlas.jets.nnjvt_update", 1,
      [](const PluginConfig& config, const PluginContext& context) {
        config.require_keys({"max_pt", "cut_map", "mev_per_gev", "network_json"});
        check_children(context, 4, 0);
        return std::make_shared<NnJvtUpdate>(config, context);
      });
  registry.register_node("org.atlas.jets.nnjvt_select", 1,
      [](const PluginConfig& config, const PluginContext& context) {
        config.require_keys({"min_pt", "max_pt", "min_abs_eta", "max_abs_eta", "cut_map", "mev_per_gev"});
        check_children(context, 2, 0);
        return std::make_shared<NnJvtSelect>(config, context);
      });
  registry.register_node("org.atlas.jets.nnjvt_sf", 1,
      [](const PluginConfig& config, const PluginContext& context) {
        config.require_keys({"min_pt", "max_pt", "min_abs_eta", "max_abs_eta"});
        check_children(context, 3, 5);
        return std::make_shared<NnJvtSf>(config, context);
      });
}
