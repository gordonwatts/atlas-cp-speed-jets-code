#include "event_inputs.h"
#include "truth_context.h"

#include <correction_plugin.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Int = std::int64_t;
using Var = correction::Variable::VarType;
using Values = std::vector<correction::Variable::Type>;
using Cfg = correction::PluginConfig;
using Ctx = correction::PluginContext;

thread_local std::string last_error;
template<class F> int checked(F fn) noexcept {
  try { fn(); last_error.clear(); return 0; }
  catch (const std::exception& e) { last_error = e.what(); return 1; }
}

class Ranges {
public:
  Ranges(const Int* offsets, size_t count, size_t rows) : offsets_(offsets), count_(count) {
    if (!count || !offsets || offsets[0] != 0 || offsets[count-1] < 0 ||
        static_cast<size_t>(offsets[count-1]) != rows || !std::is_sorted(offsets, offsets+count))
      throw std::invalid_argument("invalid jagged offsets");
  }
  std::pair<size_t,size_t> at(Int event) const {
    if (event < 0 || static_cast<size_t>(event) >= count_-1) throw std::out_of_range("event index");
    return {static_cast<size_t>(offsets_[event]),static_cast<size_t>(offsets_[event+1])};
  }
private:
  const Int* offsets_;
  size_t count_;
};

struct Vertices {
  Ranges ranges;
  const std::int32_t* types;
  const std::int32_t* tracks;
};
struct Weights { Ranges ranges; const double* values; };
struct Truth {
  std::vector<size_t> offsets;
  std::vector<jets::P4> data;
  std::unique_ptr<jets::TruthContext> lease;
};
thread_local std::unique_ptr<Vertices> vertices;
thread_local std::unique_ptr<Weights> weights;
thread_local std::unique_ptr<Truth> truth;
#ifdef JETS_LARGE_JMR_MATCH_CACHE
thread_local std::unique_ptr<jets::TruthContext> jmr_truth;
#endif

class EventInputs final : public correction::PluginNode {
public:
  EventInputs(const Cfg& cfg, const Ctx& ctx)
    : event_(ctx.input_index("event",Var::integer)), pv_(ctx.input_index("pv_index",Var::integer)),
      has_pv_(ctx.input_index("pv_index_present",Var::integer)), npv_(ctx.input_index("npv",Var::real)),
      has_npv_(ctx.input_index("npv_present",Var::integer)), primary_(cfg.integer("primary")),
      pileup_(cfg.integer("pileup")), dummy_(cfg.integer("dummy")), min_tracks_(cfg.integer("min_tracks")) {
    cfg.require_keys({"primary","pileup","dummy","min_tracks"});
    if (ctx.output_size()!=5 || ctx.child_count()) throw std::invalid_argument("event inputs shape");
  }
  size_t output_size() const override { return 5; }
  double evaluate(const Values&) const override { throw std::logic_error("event inputs are array valued"); }
  void evaluate_array(const Values& v,double* out,size_t n) const override {
    if (!vertices || !out || n<5) throw std::logic_error("unbound vertex input or invalid output");
    auto [start,stop] = vertices->ranges.at(std::get<Int>(v.at(event_)));
    Int first = -1, nominal = 0, fallback = 0;
    for (size_t i=start;i<stop;++i) {
      if (vertices->types[i]==primary_ && first<0) first=static_cast<Int>(i-start);
      nominal += vertices->types[i]==primary_ || vertices->types[i]==pileup_;
      fallback += vertices->tracks[i]>=min_tracks_;
    }
    Int gsc=std::get<Int>(v.at(has_pv_)) ? std::get<Int>(v.at(pv_)) : first;
    if (gsc<0 || static_cast<size_t>(gsc)>=stop-start) gsc=0;
    const bool is_dummy=stop-start==1 && vertices->types[start]==dummy_;
    out[0]=nominal;
    out[1]=std::get<Int>(v.at(has_npv_)) ? std::get<double>(v.at(npv_)) : fallback;
    out[2]=gsc; out[3]=first>=0 ? first : (is_dummy ? 0 : -1);
    out[4]=first>=0 ? 2 : (is_dummy ? 1 : 0);
  }
private:
  size_t event_,pv_,has_pv_,npv_,has_npv_;
  Int primary_,pileup_,dummy_,min_tracks_;
};

class TruthLabels final : public correction::PluginNode {
public:
  TruthLabels(const Cfg& cfg,const Ctx& ctx)
    : event_(ctx.input_index("event",Var::integer)), hs_dr_(cfg.real("hs_dr")), pu_dr_(cfg.real("pu_dr")),
      hs_pt_(cfg.real("hs_pt")), pu_pt_(cfg.real("pu_pt")) {
    cfg.require_keys({"hs_dr","pu_dr","hs_pt","pu_pt"});
    if (ctx.output_size()!=2 || ctx.child_count()) throw std::invalid_argument("truth label shape");
    const std::array<const char*,4> names{"pt","eta","phi","m"};
    for(size_t i=0;i<names.size();++i) idx_[i]=ctx.input_index(names[i],Var::real);
  }
  size_t output_size() const override { return 2; }
  double evaluate(const Values&) const override { throw std::logic_error("truth labels are array valued"); }
  void evaluate_array(const Values& v,double* out,size_t n) const override {
    if (!out || n<2) throw std::invalid_argument("truth label output");
    auto get=[&](size_t i){return std::get<double>(v.at(idx_[i]));};
    const jets::P4 p(get(0),get(1),get(2),get(3));
    const auto cartesian = jets::as_cartesian(p);
    const auto group=jets::TruthContext::event(std::get<Int>(v.at(event_)));
    bool hs=false,pu=true;
    for(size_t i=0;i<group.size;++i){
      const auto& t=group.data[i];
      const float distance=jets::truth_distance(cartesian,t);
      if(distance<hs_dr_ && t.Pt()>hs_pt_) hs=true;
      if(distance<pu_dr_ && t.Pt()>pu_pt_) pu=false;
      if(hs && !pu) break;
    }
    out[0]=hs; out[1]=pu;
  }
private:
  std::array<size_t,4> idx_{};
  size_t event_;
  float hs_dr_,pu_dr_,hs_pt_,pu_pt_;
};

class LargeSelection final : public correction::PluginNode {
public:
  LargeSelection(const Cfg& cfg,const Ctx& ctx):
    low_pt_(cfg.real("min_pt")),high_pt_(cfg.real("max_pt")),max_y_(cfg.real("max_rapidity")),
    low_m_(cfg.real("min_mass")),high_m_(cfg.real("max_mass")) {
    cfg.require_keys({"min_pt","max_pt","max_rapidity","min_mass","max_mass"});
    if(ctx.output_size()!=3 || ctx.child_count()) throw std::invalid_argument("large selection shape");
    const std::array<const char*,4> names{"pt","eta","phi","m"};
    for(size_t i=0;i<names.size();++i) idx_[i]=ctx.input_index(names[i],Var::real);
  }
  size_t output_size() const override {return 3;}
  double evaluate(const Values&) const override {throw std::logic_error("selection is array valued");}
  void evaluate_array(const Values& v,double* out,size_t n) const override {
    if(!out || n<3) throw std::invalid_argument("large selection output");
    auto get=[&](size_t i){return std::get<double>(v.at(idx_[i]));};
    const float pt=get(0), mass=get(3);
    const float abs_y=std::abs(jets::P4(pt,get(1),get(2),mass).Rapidity());
    constexpr auto accept=std::numeric_limits<std::uint32_t>::max();
    auto kin=accept, omass=accept;
    if(!(std::isfinite(pt) && pt>=low_pt_)) kin &= ~1u;
    if(!(pt<high_pt_)) kin &= ~2u;
    if(!(abs_y<=max_y_)) kin &= ~4u;
    if(kin!=accept || !(std::isfinite(mass) && mass>=low_m_)) omass &= ~1u;
    if(kin!=accept || !(mass<high_m_)) omass &= ~2u;
    out[0]=kin; out[1]=omass; out[2]=kin==accept && omass==accept;
  }
private:
  std::array<size_t,4> idx_{};
  float low_pt_,high_pt_,max_y_,low_m_,high_m_;
};

class EventSF final : public correction::PluginNode {
public:
  explicit EventSF(size_t event):event_(event){}
  double evaluate(const Values& v) const override {
    if(!weights) throw std::logic_error("unbound event weights");
    auto [start,stop]=weights->ranges.at(std::get<Int>(v.at(event_)));
    float product=1;
    for (size_t i=start;i<stop;++i) product*=static_cast<float>(weights->values[i]);
    return product;
  }
private:
  size_t event_;
};
}

extern "C" const char* jets_buffer_error() {return last_error.c_str();}
extern "C" int jets_bind_truth(const Int* offsets,size_t count,const double* pt,const double* eta,
                               const double* phi,const double* mass,size_t rows) {
  return checked([&]{
    if(truth || (rows && (!pt || !eta || !phi || !mass))) throw std::invalid_argument("invalid truth bind");
    Ranges validated(offsets,count,rows);
    auto value=std::make_unique<Truth>();
    value->offsets.assign(offsets,offsets+count);
    value->data.reserve(rows);
    for(size_t i=0;i<rows;++i) value->data.emplace_back(pt[i],eta[i],phi[i],mass[i]);
    value->lease=std::make_unique<jets::TruthContext>(value->data.data(),rows,value->offsets.data(),count);
    truth=std::move(value);
  });
}
#ifdef JETS_LARGE_JMR_MATCH_CACHE
extern "C" int jets_bind_jmr_truth(const Int* offsets, size_t count, const double* pt,
                                    const double* eta, const double* phi, const double* mass, size_t rows) {
  return checked([&] {
    jmr_truth = jets::TruthContext::own_jmr(offsets, count, pt, eta, phi, mass, rows);
  });
}
#endif
extern "C" int jets_bind_vertices(const Int* offsets,size_t count,const std::int32_t* types,
                                  const std::int32_t* tracks,size_t rows) {
  return checked([&]{
    if(vertices || (rows && (!types || !tracks))) throw std::invalid_argument("invalid vertex bind");
    vertices=std::make_unique<Vertices>(Vertices{Ranges(offsets,count,rows),types,tracks});
  });
}
extern "C" int jets_bind_weights(const Int* offsets,size_t count,const double* data,size_t rows) {
  return checked([&]{
    if(weights || (rows && !data)) throw std::invalid_argument("invalid weights bind");
    weights=std::make_unique<Weights>(Weights{Ranges(offsets,count,rows),data});
  });
}
extern "C" int jets_unbind_truth() {
  return checked([] {
#ifdef JETS_LARGE_JMR_MATCH_CACHE
    if (jmr_truth) {
      jmr_truth.reset();
      return;
    }
#endif
    if (!truth) throw std::logic_error("unbound truth");
    truth.reset();
  });
}
extern "C" int jets_unbind_vertices(){return checked([]{if(!vertices) throw std::logic_error("unbound vertices"); vertices.reset();});}
extern "C" int jets_unbind_weights(){return checked([]{if(!weights) throw std::logic_error("unbound weights"); weights.reset();});}

void register_event_inputs(correction::PluginRegistry& r) {
  r.register_node("org.atlas.jets.event_inputs",1,[](const auto& c,const auto& x){return std::make_shared<EventInputs>(c,x);});
  r.register_node("org.atlas.jets.pileup_truth",1,[](const auto& c,const auto& x){return std::make_shared<TruthLabels>(c,x);});
  r.register_node("org.atlas.jets.large_selection",1,[](const auto& c,const auto& x){return std::make_shared<LargeSelection>(c,x);});
  r.register_node("org.atlas.jets.event_sf",1,[](const auto& c,const auto& x){
    c.require_keys({}); if(x.output_size() || x.child_count()) throw std::invalid_argument("event SF shape");
    return std::make_shared<EventSF>(x.input_index("event",Var::integer));});
}
