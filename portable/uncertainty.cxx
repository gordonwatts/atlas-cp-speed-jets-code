// MC23 JetUncertainties, derived from the AnalysisBase 25.2.117 source.
// Copyright (C) 2002-2026 CERN for the benefit of the ATLAS collaboration.
#include "../provider/jet_uncertainty.h"
#include "../provider/jet_common.h"
#include "../provider/jet_smearing.h"
#include "histogram.h"
#ifdef JETS_NOMINAL_RESOLUTION_CACHE
#include "nominal_resolution_cache.h"
#endif

#include <correction_plugin.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace jets {
namespace {
using Json = rapidjson::Value;
using portable::Interpolation;
using portable::UncertaintyLookup;
constexpr double error_value = -1234.0; // JetUncertainties/Helpers.h

enum class Param { pt, pt_eta, pt_abs_eta, pt_mass, energy_log_mass_abs_eta };
enum class Kind { table, npv, mu, pt_term, rho, composition, response, bjes, topology };
enum class Scale { four_vector, mass, pt_resolution, four_vector_resolution };
template<class T> T parse(const std::string& value, std::initializer_list<std::pair<const char*,T>> values) {
  for (const auto& [name, code] : values) if (value == name) return code;
  throw std::invalid_argument("Unsupported jet model value: " + value);
}

bool valid(double number) { return number < 1.0+1.e-5 && number > 1.0-1.e-5; }

struct Component {
  Param param;
  Kind kind;
  Scale scale;
  bool calo;
  std::vector<int> labels;
  std::vector<const UncertaintyLookup*> tables;
  const UncertaintyLookup *validity=nullptr, *fraction=nullptr, *fraction_error=nullptr;
  bool first_mu=false, first_gluon=false;
};

struct Group {
  bool quadrature;
  std::map<Scale, std::vector<Component>> pieces;
};

struct State { const Group* group; int sign; bool pseudo; };

struct Coordinates {
  const JetUncertaintyInput& input;
  const P4 main, calo;
  double pt, eta, energy;
  explicit Coordinates(const JetUncertaintyInput& in, float scale)
    :input(in),main(vector(in.p4)),calo(in.calo_scale ? vector(*in.calo_scale) : main),
     pt(in.p4[0]*scale),eta(in.p4[1]),energy(main.E()*scale) {}
  static P4 vector(const std::array<double,4>& p) { return P4(p[0],p[1],p[2],p[3]); }
  double mass(bool use_calo) const {
    return use_calo && input.calo_scale ? calo.M() : input.p4[3];
  }
  double mass_over_energy(bool use_calo) const {
    return mass(use_calo)/(use_calo && input.calo_scale ? calo.E() : main.E());
  }
};

double lookup(const UncertaintyLookup& hist, const Component& component, const Coordinates& c, float scale) {
  switch (component.param) {
    case Param::pt: return hist.getValue(c.pt);
    case Param::pt_eta: return hist.getValue(c.pt,c.eta);
    case Param::pt_abs_eta: return hist.getValue(c.pt,std::abs(c.eta));
    case Param::pt_mass: return hist.getValue(c.pt,c.mass(component.calo)*scale);
    case Param::energy_log_mass_abs_eta:
      return hist.getValue(c.energy,std::log(c.mass_over_energy(component.calo)),std::abs(c.eta));
  }
  throw std::logic_error("Invalid component parametrization");
}
}

struct JetUncertaintyModel::Impl {
  using LookupKey = std::pair<std::string,Interpolation>;
  std::map<LookupKey,UncertaintyLookup> lookups;
  std::map<std::string,Group> groups;
  std::map<std::string,State> states;
  const UncertaintyLookup *file_validity=nullptr,*res_mc=nullptr,*res_data=nullptr;
  float energy_scale, ref_npv, ref_mu;
  bool smear_only_mc, abs_fraction;
  ResolutionRandom random;
#ifdef JETS_NOMINAL_RESOLUTION_CACHE
  const std::uint64_t nominal_resolution_identity = portable::nominal_resolution_identity();
#endif

  explicit Impl(const correction::PluginConfig& config) {
    rapidjson::Document p;
    p.Parse<rapidjson::kParseFullPrecisionFlag>(config.json().c_str());
    if(p.HasParseError() || !p.HasMember("portable_plan")) throw std::invalid_argument("Missing portable jet model");
    const auto& plan=p["portable_plan"];
    energy_scale=static_cast<float>(config.real("mev_to_gev"));
    ref_mu=static_cast<float>(plan["ref_mu"].GetDouble());
    ref_npv=static_cast<float>(plan["ref_npv"].GetDouble());
    smear_only_mc=plan["smear_only_mc"].GetBool();
    abs_fraction=p["abs_eta_gluon_fraction"].GetBool();
    random={config.integer("rng_user_seed"),config.real("mc_phi_seed_multiplier"),
            config.real("pseudo_data_phi_seed_multiplier"),config.integer("mc_zero_seed"),
            config.integer("pseudo_data_zero_seed")};
    auto table=[&](const std::string& name,const std::string& mode,bool analysis=false) -> const UncertaintyLookup* {
      Interpolation interp=parse<Interpolation>(mode,{{"Full",Interpolation::Full},{"None",Interpolation::None},{"OnlyX",Interpolation::OnlyX}});
      auto key=LookupKey{name,interp};
      auto existing=lookups.find(key);
      if(existing!=lookups.end()) return &existing->second;
      const auto& store=p[analysis ? "analysis_histograms" : "histograms"];
      if(!store.HasMember(name.c_str())) throw std::invalid_argument("Missing embedded jet table: "+name);
      return &lookups.try_emplace(key,portable::restore_histogram(store[name.c_str()]),interp).first->second;
    };
    for(const auto& group:plan["groups"].GetObject()) {
      const auto& g=group.value;
      auto& dest=groups[group.name.GetString()];
      dest.quadrature=std::string(g["correlation"].GetString())=="Uncorrelated";
      for(const auto& comp:g["components"].GetArray()) {
        Component c;
        c.param=parse<Param>(comp["param"].GetString(),{{"Pt",Param::pt},{"PtEta",Param::pt_eta},{"PtAbsEta",Param::pt_abs_eta},
                    {"PtAbsMass",Param::pt_mass},{"eLOGmOeAbsEta",Param::energy_log_mass_abs_eta}});
        c.kind=parse<Kind>(comp["kind"].GetString(),{{"table",Kind::table},{"pileup_offsetnpv",Kind::npv},{"pileup_offsetmu",Kind::mu},
                    {"pileup_ptterm",Kind::pt_term},{"pileup_rhotopology",Kind::rho},{"flavor_composition",Kind::composition},
                    {"flavor_response",Kind::response},{"bjes_response",Kind::bjes},{"larger_topologyuncertainty",Kind::topology}});
        c.scale=parse<Scale>(comp["scale"].GetString(),{{"fourvec",Scale::four_vector},{"mass",Scale::mass},
                    {"ptresabs",Scale::pt_resolution},{"fourvecresabs",Scale::four_vector_resolution}});
        c.calo=comp["calo"].GetBool();
        for(const auto& label:comp["labels"].GetArray()) c.labels.push_back(label.GetInt());
        const auto mode=std::string(comp["mode"].GetString());
        for(const auto& name:comp["tables"].GetArray()) c.tables.push_back(table(name.GetString(),mode));
        const std::string first=comp["tables"][0].GetString();
        c.first_mu=first.find("_mu_")!=std::string::npos;
        c.first_gluon=first.find("Glu")!=std::string::npos;
        std::string valid_name=comp["validity"].GetString();
        if(!valid_name.empty()) c.validity=table(valid_name,"None");
        if(c.kind==Kind::composition || c.kind==Kind::response) {
          c.fraction=table(p["gluon_fraction_key"].GetString(),mode,true);
          c.fraction_error=table(p["gluon_fraction_error_key"].GetString(),mode,true);
        }
        dest.pieces[c.scale].push_back(std::move(c));
      }
    }
    for(const auto& entry:p["states"].GetObject()) {
      const auto& s=entry.value;
      states.emplace(entry.name.GetString(),State{&groups.at(s["basename"].GetString()),s["parameter"].GetInt(),s["pseudo_data"].GetBool()});
    }
    std::string validity=plan["file_validity"].GetString();
    if(!validity.empty()) file_validity=table(validity,"None");
    for(const auto& entry:plan["resolutions"].GetObject()) {
      const auto* hist=table(entry.value["table"].GetString(),entry.value["mode"].GetString());
      if(std::string(entry.name.GetString()).ends_with("MC")) res_mc=hist; else res_data=hist;
    }
    if(!res_mc || !res_data) throw std::invalid_argument("Missing nominal jet resolutions");
  }

  double value(const Component& c,const Coordinates& coords,bool bjet) const {
    if(!c.labels.empty()) {
      int label=coords.input.large_r_truth_label.value_or(0);
      if(label==0 || ((c.kind==Kind::response || c.kind==Kind::composition) && (label<1 || label>19))) return error_value;
      if(std::find(c.labels.begin(),c.labels.end(),label)==c.labels.end()) return 0.0;
    }
    const auto lookup_at=[&](size_t i){return lookup(*c.tables.at(i),c,coords,energy_scale);};
    const auto pileup_weight=[&](Kind kind) -> double {
      if(coords.input.uncertainty_npv<=0) return error_value;
      if(kind==Kind::mu) return coords.input.average_mu - static_cast<double>(ref_mu);
      if(kind==Kind::npv) return coords.input.uncertainty_npv - static_cast<double>(ref_npv);
      return 1.0;
    };
    switch(c.kind) {
      case Kind::table: case Kind::topology: return lookup_at(0);
      case Kind::bjes: return bjet ? lookup_at(0) : 0.0;
      case Kind::mu: case Kind::npv: case Kind::rho: return pileup_weight(c.kind)*lookup_at(0);
      case Kind::pt_term: {
        double result=pileup_weight(c.first_mu ? Kind::mu : Kind::npv)*lookup_at(0);
        result+=pileup_weight(c.first_mu ? Kind::npv : Kind::mu)*lookup_at(1);
        return result;
      }
      case Kind::response: case Kind::composition: {
        if(bjet) return 0.0;
        double eta=(abs_fraction || c.param==Param::pt_abs_eta) ? std::abs(coords.eta) : coords.eta;
        double fraction=c.fraction->getValue(coords.pt,eta);
        if(c.kind==Kind::response) return lookup_at(0)*fraction;
        double gluon=lookup_at(c.first_gluon ? 0 : 1), quark=lookup_at(c.first_gluon ? 1 : 0);
        double sample=fraction*gluon+(1-fraction)*quark;
        return sample==0.0 ? error_value : c.fraction_error->getValue(coords.pt,eta)*std::abs(quark-gluon)/sample;
      }
    }
    throw std::logic_error("Invalid component type");
  }

  const State* select(const std::string& systematic) const {
    const auto found=states.find(systematic);
    return found==states.end() ? nullptr : &found->second;
  }

  JetUncertaintyResult evaluate(const JetUncertaintyInput& input,const State* state) const {
    bool pseudo=state && state->pseudo;
    JetUncertaintyResult out{input.p4,std::nullopt,std::nullopt};
    if(!pseudo) out.is_bjet=input.parton_truth_label==5;
    if(!input.preselected) return out;
    out.status=JetUncertaintyStatus::error;
    if(!input.event_available) return out;
    Coordinates coords(input,energy_scale);
    out.status=JetUncertaintyStatus::out_of_validity;
    if(file_validity && !valid(file_validity->getValue(coords.pt,coords.mass(true)*energy_scale,coords.eta))) return out;
    if(state) {
      const auto& group=*state->group;
      for(const auto& [scale,pieces]:group.pieces) {
        for(const auto& c:pieces) {
          if(c.param==Param::energy_log_mass_abs_eta && !(coords.mass_over_energy(c.calo)>0)) return out;
          if(c.validity && !valid(lookup(*c.validity,c,coords,energy_scale))) return out;
        }
      }
      for(const auto& [scale,pieces]:group.pieces) {
        double uncertainty=0.0;
        for(const auto& c:pieces) {
          double v=value(c,coords,out.is_bjet.value_or(false));
          uncertainty+=group.quadrature && pieces.size()>1 ? std::pow(v,2) : v;
        }
        if(group.quadrature && pieces.size()>1) uncertainty=std::sqrt(uncertainty);
        uncertainty*=state->sign;
        double pt_factor=1.0, mass_factor=1.0;
        if(scale==Scale::four_vector) pt_factor=mass_factor=1.0+uncertainty;
        else if(scale==Scale::mass) mass_factor=1.0+uncertainty;
        else {
          const auto scalar_nominal=[&](bool mc) {
            return (mc?res_mc:res_data)->getValue(out.p4[0]*energy_scale,std::abs(out.p4[1]));
          };
#ifdef JETS_NOMINAL_RESOLUTION_CACHE
          portable::ProductionNominalResolutionCache::Query resolution_query;
          auto nominal=[&](bool mc) {
            auto& cache = portable::ProductionNominalResolutionCache::local();
            if (mc)
              return cache.mc(resolution_query, nominal_resolution_identity, out.p4[0], out.p4[1],
                              [&] { return scalar_nominal(true); });
            return cache.data(resolution_query, [&] { return scalar_nominal(false); });
          };
#else
          const auto& nominal=scalar_nominal;
#endif
          pt_factor=resolution_factor(out.p4[2],uncertainty,pseudo,smear_only_mc,false,random,nominal);
          if(scale==Scale::four_vector_resolution) mass_factor=pt_factor;
        }
        const auto& p=out.p4;
        out.p4=stored_coordinates(relative_pt_mass(p[0],p[1],p[2],p[3],pt_factor,mass_factor));
      }
    }
    out.status=JetUncertaintyStatus::ok;
    return out;
  }
};
JetUncertaintyModel::JetUncertaintyModel(const correction::PluginConfig& config):model_(std::make_unique<Impl>(config)) {}
JetUncertaintyModel::~JetUncertaintyModel()=default;
JetUncertaintyResult JetUncertaintyModel::evaluate(const JetUncertaintyInput& input,const std::string& systematic) const {
  return model_->evaluate(input,model_->select(systematic));
}
#ifdef JETS_UNCERTAINTY_BATCH
JetUncertaintyModel::Selected JetUncertaintyModel::select(const std::string& systematic) const {
  const State* state=model_->select(systematic);
  return [this,state](const JetUncertaintyInput& input){ return model_->evaluate(input,state); };
}
#endif
} // namespace jets
