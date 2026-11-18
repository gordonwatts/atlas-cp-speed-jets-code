// Active spline-response prescription from ATLAS JetCalibTools, EtaJESCorrection.
// Copyright (C) 2002-2026 CERN for the benefit of the ATLAS collaboration.
#include "eta_jes.h"
#include "histogram.h"
#include "jet_common.h"

#include <TLorentzVector.h>
#include <correction_plugin.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
using Value = rapidjson::Value;
using Variables = std::vector<correction::Variable::Type>;

std::vector<double> read_numbers(const Value& values) {
  if (!values.IsArray()) throw std::invalid_argument("EtaJES expected numeric array");
  std::vector<double> result;
  for (const auto& value : values.GetArray()) {
    if (!value.IsNumber() || !std::isfinite(value.GetDouble()))
      throw std::invalid_argument("invalid EtaJES number");
    result.push_back(value.GetDouble());
  }
  return result;
}

struct EtaBin {
  std::unique_ptr<TH1> response;
  std::vector<double> direction;
  double freeze_at, cutoff, cutoff_response, slope;
};

class EtaJES final : public correction::PluginNode {
public:
  EtaJES(const correction::PluginConfig& config, const correction::PluginContext& ctx) {
    config.require_keys({"eta_edges", "bins", "min_pt_gev", "use_secondary", "secondary_abs_eta",
                         "secondary_pt_gev", "eta_min_pt_gev", "eta_max_energy_gev", "mev_per_gev"});
    if (ctx.output_size() != 4 || ctx.child_count()) throw std::invalid_argument("EtaJES output shape");
    const std::array<const char*,5> names{"pt","eta","phi","m","detector_eta"};
    for (size_t i=0; i<names.size(); ++i)
      index_[i] = ctx.input_index(names[i], correction::Variable::VarType::real);
    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseFullPrecisionFlag>(config.json().c_str());
    if (doc.HasParseError() || !doc["use_secondary"].IsBool()) throw std::invalid_argument("EtaJES payload");
    edges_ = read_numbers(doc["eta_edges"]);
    if (edges_.size() < 2 || std::adjacent_find(edges_.begin(), edges_.end(),
          std::greater_equal<double>()) != edges_.end()) throw std::invalid_argument("EtaJES eta edges");
    const auto& rows = doc["bins"];
    if (!rows.IsArray() || rows.Size() != edges_.size()-1) throw std::invalid_argument("EtaJES rows");
    min_pt_ = config.real("min_pt_gev");
    secondary_ = doc["use_secondary"].GetBool();
    secondary_abs_eta_ = config.real("secondary_abs_eta");
    secondary_pt_ = config.real("secondary_pt_gev");
    eta_min_pt_ = config.real("eta_min_pt_gev");
    eta_max_e_ = config.real("eta_max_energy_gev");
    unit_ = config.real("mev_per_gev");
    for (size_t i = 0; i < rows.Size(); ++i) {
      const auto& row = rows[static_cast<unsigned>(i)];
      if (!row.IsObject() || !row.HasMember("response") || !row.HasMember("eta_coefficients") ||
          !row.HasMember("freeze_at") || !row["freeze_at"].IsNumber()) throw std::invalid_argument("EtaJES table");
      auto hist = jets::restore_histogram(row["response"]);
      auto direction = read_numbers(row["eta_coefficients"]);
      if (hist->GetDimension()!=1 || direction.empty()) throw std::invalid_argument("EtaJES dimensions");
      const double cutoff = minimum_pt(edges_[i]) * std::cosh(edges_[i]);
      const int b = hist->FindBin(cutoff);
      const double r0 = hist->Interpolate(cutoff);
      const double slope = (hist->GetBinContent(b+1)-hist->GetBinContent(b)) /
                           (hist->GetBinCenter(b+1)-hist->GetBinCenter(b));
      bins_.push_back({std::move(hist), std::move(direction), row["freeze_at"].GetDouble(), cutoff, r0, slope});
    }
  }

  size_t output_size() const override { return 4; }
  double evaluate(const Variables&) const override { throw std::logic_error("EtaJES returns a four-vector"); }

  jets::P4 correct(const jets::P4& input, float detector_eta) const {
    const double eta = detector_eta;
    const double cosh_eta = std::cosh(eta);
    const double energy = input.E() / unit_;
    auto upper = std::upper_bound(edges_.begin(), edges_.end(), eta);
    const auto i = std::clamp<std::ptrdiff_t>(upper - edges_.begin() - 1, 0, bins_.size()-1);
    const auto& bin = bins_[i];
    double jes;
    if (energy / cosh_eta < minimum_pt(eta)) {
      const double response = bin.slope * (energy - bin.cutoff) + bin.cutoff_response;
      jes = response == 0.0 ? 0.0 : 1.0/response;
    } else {
      const double e = bin.freeze_at != -1.0 && energy > bin.freeze_at ? bin.freeze_at : energy;
      jes = 1.0 / bin.response->Interpolate(e);
    }
    auto scaled = input * jes;
    const double corrected_gev = std::min(std::max(scaled.E()/unit_, eta_min_pt_*cosh_eta), eta_max_e_);
    double offset = 0.0;
    for (size_t k = 0; k < bin.direction.size(); ++k)
      offset += bin.direction[k]*std::pow(std::log(corrected_gev), static_cast<int>(k));
    const double direction = scaled.Eta() + (-offset);
    TLorentzVector cartesian;
    cartesian.SetPtEtaPhiM(scaled.P()/std::cosh(direction), direction, scaled.Phi(), scaled.M());
    scaled.SetPxPyPzE(cartesian.Px(), cartesian.Py(), cartesian.Pz(), cartesian.E());
    return jets::stored_p4(scaled);
  }

  void evaluate_array(const Variables& values, double* out, size_t n) const override {
    if (!out || n < 4) throw std::invalid_argument("EtaJES output capacity");
    const auto get = [&](size_t i) { return std::get<double>(values.at(index_[i])); };
    const auto p4 = correct(jets::P4(get(0),get(1),get(2),get(3)), static_cast<float>(get(4)));
    out[0]=p4.Pt(); out[1]=p4.Eta(); out[2]=p4.Phi(); out[3]=p4.M();
  }

private:
  double minimum_pt(double eta) const {
    return secondary_ && std::abs(eta)>=secondary_abs_eta_ ? secondary_pt_ : min_pt_;
  }
  std::array<size_t,5> index_{};
  std::vector<double> edges_;
  std::vector<EtaBin> bins_;
  double min_pt_, secondary_abs_eta_, secondary_pt_, eta_min_pt_, eta_max_e_, unit_;
  bool secondary_;
};
}

void register_eta_jes(correction::PluginRegistry& registry) {
  registry.register_node("org.atlas.jets.eta_jes", 1, [](const auto& config, const auto& context) {
    return std::make_shared<EtaJES>(config, context);
  });
}
