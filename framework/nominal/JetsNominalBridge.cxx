#include "JetsNominalBridge.h"

#include "JetCalibTools/JetEventInfo.h"
#include "JetsCorrectionRuntime/JetIO.h"

#include <AthContainers/Accessor.h>
#include <AthContainers/ConstAccessor.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {
using jets::framework::bind;
using jets::framework::CorrectionCall;
using jets::framework::Values;
using jets::framework::MomentumAux;
using P4 = std::array<double, 4>;

constexpr std::string_view small_definition = "AntiKt4EMPFlow";
constexpr std::string_view large_definition = "AntiKt10UFOCSSKSoftDropBeta100Zcut10";
constexpr std::string_view small_config =
    "AntiKt4EMPFlow_MC23a_PreRecR22_Phase2_CalibConfig_ResPU_EtaJES_GSC_241208_InSitu.config";
constexpr std::string_view large_config = "JES_MC20PreRecommendation_R10_UFO_CSSK_SoftDrop_JMS_R21Insitu_26Nov2024.config";

const MomentumAux assigned("");
const MomentumAux constituent("JetConstitScaleMomentum_");
const MomentumAux pileup("JetPileupScaleMomentum_");
const MomentumAux eta_jes("JetEtaJESScaleMomentum_");
const MomentumAux jms("JetJMSScaleMomentum_");
const MomentumAux gsc("JetGSCScaleMomentum_");
const SG::ConstAccessor<float> detector_eta("DetectorEta");
const SG::ConstAccessor<float> area_pt("ActiveArea4vec_pt");

const CorrectionCall& pileup_call() {
  static const auto& call = bind("small_pileup", 4,
      {"pt", "eta", "phi", "m", "area_pt", "rho", "mu", "npv"});
  return call;
}
const CorrectionCall& eta_jes_call(bool small) {
  if (small) {
    static const auto& call = bind("small_eta_jes", 4, {"pt", "eta", "phi", "m", "detector_eta"});
    return call;
  }
  static const auto& call = bind("large_eta_jes", 4, {"pt", "eta", "phi", "m", "detector_eta"});
  return call;
}
const CorrectionCall& jms_call() {
  static const auto& call = bind("large_jms", 4, {"pt", "eta", "phi", "m", "detector_eta"});
  return call;
}
const CorrectionCall& gsc_call() {
  static const auto& call = bind("small_gsc", 4,
      {"pt", "eta", "phi", "m", "detector_eta", "constituent_pt", "constituent_eta",
       "constituent_phi", "constituent_m", "sum_pt_charged_pfo", "n_trk", "track_width",
       "em_b3", "em_e3", "tile_bar0", "tile_ext0"});
  return call;
}

void evaluate_and_write(const CorrectionCall& call, const Values& values,
                        const MomentumAux& stage, xAOD::Jet& jet) {
  P4 output;
  call.array(values, output.data(), output.size());
  stage.write(jet, output.data());
  assigned.write(jet, output.data());
}
}  // namespace

namespace jets::framework {

void validate_nominal(std::string_view definition, std::string_view config,
                      std::string_view sequence, std::string_view area,
                      bool standard_properties) {
  if (standard_properties && area == "00-04-83") {
    if (definition == small_definition && config == small_config &&
        sequence == "JetArea_Residual_EtaJES_GSC") {
      pileup_call();
      eta_jes_call(true);
      gsc_call();
      return;
    }
    if (definition == large_definition && config == large_config && sequence == "EtaJES_JMS") {
      eta_jes_call(false);
      jms_call();
      return;
    }
  }
  throw std::runtime_error("unsupported framework nominal recipe: " + std::string(definition) +
                           " " + std::string(config) + " " + std::string(sequence));
}

void apply_small_pileup(xAOD::Jet& jet, JetEventInfo& event) {
  Values values;
  values.reserve(8);
  constituent.append(jet, values);
  values.emplace_back(static_cast<double>(area_pt(jet)));
  values.emplace_back(event.rho());
  values.emplace_back(event.mu());
  values.emplace_back(static_cast<int64_t>(event.NPV()));
  evaluate_and_write(pileup_call(), values, pileup, jet);
  jet.setAttribute<int>("OriginCorrected", false);
  jet.setAttribute<int>("PileupCorrected", true);
}

void apply_eta_jes(xAOD::Jet& jet, std::string_view definition, std::string_view starting_scale) {
  const bool small = definition == small_definition;
  if ((!small && definition != large_definition) ||
      starting_scale != (small ? "JetPileupScaleMomentum" : "JetConstitScaleMomentum"))
    throw std::runtime_error("unexpected EtaJES framework starting scale");
  Values values;
  values.reserve(5);
  (small ? pileup : constituent).append(jet, values);
  values.emplace_back(static_cast<double>(detector_eta(jet)));
  evaluate_and_write(eta_jes_call(small), values, eta_jes, jet);
}

void apply_large_jms(xAOD::Jet& jet) {
  Values values;
  values.reserve(5);
  eta_jes.append(jet, values);
  values.emplace_back(static_cast<double>(detector_eta(jet)));
  evaluate_and_write(jms_call(), values, jms, jet);
}

void apply_small_gsc(xAOD::Jet& jet, JetEventInfo& event) {
  static const SG::ConstAccessor<std::vector<float>> charged("SumPtChargedPFOPt500");
  static const SG::ConstAccessor<std::vector<float>> width("TrackWidthPt1000");
  static const SG::ConstAccessor<std::vector<int>> tracks("NumTrkPt1000");
  static const SG::ConstAccessor<std::vector<float>> sampling("EnergyPerSampling");
  const auto vertex = static_cast<size_t>(event.PVIndex());
  Values values;
  values.reserve(16);
  eta_jes.append(jet, values);
  values.emplace_back(static_cast<double>(detector_eta(jet)));
  constituent.append(jet, values);
  values.emplace_back(static_cast<double>(charged(jet).at(vertex)));
  values.emplace_back(static_cast<int64_t>(tracks(jet).at(vertex)));
  values.emplace_back(static_cast<double>(width(jet).at(vertex)));
  const auto& energies = sampling(jet);
  for (const auto slot : {3, 7, 12, 18}) values.emplace_back(static_cast<double>(energies.at(slot)));
  evaluate_and_write(gsc_call(), values, gsc, jet);
}

}  // namespace jets::framework
