#include "jet_uncertainty.h"

#include "histogram.h"
#include "jet_common.h"
#include "jet_smearing.h"
#include "uncertainty_components.h"

#include <AthContainers/AuxElement.h>
#include <JetUncertainties/Helpers.h>
#include <JetUncertainties/ResolutionHelper.h>
#include <JetUncertainties/UncertaintySet.h>
#include <JetUncertainties/ValidityHistogram.h>
#include <PATInterfaces/SystematicSet.h>
#include <TDirectory.h>
#include <TEnv.h>
#include <TMemFile.h>
#include <correction_plugin.h>
#include <rapidjson/document.h>
#include <xAODEventInfo/EventInfo.h>
#include <xAODJet/Jet.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace {

using Json = rapidjson::Value;
using Scale = jet::CompScaleVar::TypeEnum;
using Status = jets::JetUncertaintyStatus;

const Json& field(const Json& object, const char* key) {
  if (!object.IsObject() || !object.HasMember(key)) {
    throw std::invalid_argument(std::string("Missing jet-uncertainty field: ") + key);
  }
  return object[key];
}

void require_success(const StatusCode& status, const std::string& message) {
  if (status.isFailure()) throw std::runtime_error(message);
}

void write_histograms(TMemFile& file, const Json& histograms) {
  if (!file.IsOpen() || !histograms.IsObject()) {
    throw std::invalid_argument("Invalid jet-uncertainty histogram store");
  }
  TDirectory::TContext current(&file);
  for (const auto& entry : histograms.GetObject()) {
    auto histogram = jets::restore_histogram(entry.value);
    histogram->SetName(entry.name.GetString());
    if (histogram->Write(entry.name.GetString()) <= 0) {
      throw std::runtime_error("Could not restore embedded jet-uncertainty histogram");
    }
  }
}

double nominal_resolution(const jet::ResolutionHelper& resolutions, const xAOD::Jet& jet,
                          Scale type, jet::JetTopology::TypeEnum topology,
                          bool read_mc, float energy_scale) {
  const auto [histogram, param, mass_definition] = resolutions.getNominalResolution(type, topology, read_mc);
  if (!histogram) return JESUNC_ERROR_CODE;
  if (param != jet::CompParametrization::PtAbsEta || mass_definition != jet::CompMassDef::UNKNOWN) {
    throw std::invalid_argument("Unmapped nominal resolution parametrization in the active recommendations");
  }
  const auto p4 = jet.jetP4();
  return histogram->getValue(p4.Pt() * energy_scale, std::abs(p4.Eta()));
}

double resolution_factor(const jet::ResolutionHelper& resolutions, const xAOD::Jet& calibrated,
                         Scale scale, jet::JetTopology::TypeEnum topology, double variation,
                         bool pseudo_data, float energy_scale, const jets::ResolutionRandom& random) {
  return jets::resolution_factor(calibrated.phi(), variation, pseudo_data, resolutions.smearOnlyMC(),
      jet::CompScaleVar::isRelResolutionType(scale), random, [&](bool read_mc) {
        return nominal_resolution(resolutions, calibrated, scale, topology, read_mc, energy_scale);
      });
}

void set_jet_inputs(xAOD::Jet& jet, const jets::JetUncertaintyInput& input, std::optional<bool> is_bjet) {
  static const SG::Accessor<float> pt("pt"), eta("eta"), phi("phi"), mass("m");
  static const SG::Accessor<int> truth("R10TruthLabel_R21Precision_2022v1");
  static const SG::Accessor<char> bjet("IsBjet");
  static const SG::Accessor<float> calo_pt("JetJMSScaleMomentumCalo_pt"),
      calo_eta("JetJMSScaleMomentumCalo_eta"), calo_phi("JetJMSScaleMomentumCalo_phi"),
      calo_mass("JetJMSScaleMomentumCalo_m");
  // The supplied coordinates are raw xAOD auxiliary values. Using setJetP4
  // here would normalize phi before the tool observes the input values.
  pt(jet) = static_cast<float>(input.p4[0]);
  eta(jet) = static_cast<float>(input.p4[1]);
  phi(jet) = static_cast<float>(input.p4[2]);
  mass(jet) = static_cast<float>(input.p4[3]);
  if (input.large_r_truth_label) truth(jet) = *input.large_r_truth_label;
  if (is_bjet) bjet(jet) = *is_bjet ? 1 : 0;
  if (input.calo_scale) {
    calo_pt(jet) = static_cast<float>((*input.calo_scale)[0]);
    calo_eta(jet) = static_cast<float>((*input.calo_scale)[1]);
    calo_phi(jet) = static_cast<float>((*input.calo_scale)[2]);
    calo_mass(jet) = static_cast<float>((*input.calo_scale)[3]);
  }
}

void update_coordinates(xAOD::Jet& jet, double pt_factor, double mass_factor) {
  jet.setJetP4(jets::relative_pt_mass(jet.pt(), jet.eta(), jet.phi(), jet.m(), pt_factor, mass_factor));
}

bool is_supported_scale(Scale scale) {
  using namespace jet::CompScaleVar;
  return scale == FourVec || scale == Pt || scale == Mass ||
         scale == PtRes || scale == PtResAbs || scale == FourVecRes || scale == FourVecResAbs;
}

struct ConfiguredState {
  std::unique_ptr<const jet::UncertaintySet> uncertainty;
  bool pseudo_data;
};

}  // namespace

namespace jets {

struct JetUncertaintyModel::Impl {
  explicit Impl(const correction::PluginConfig& configuration) {
    configuration.require_keys({"jet_definition", "mc_type", "group_prefix", "settings",
        "histograms", "analysis_histograms", "gluon_fraction_key", "gluon_fraction_error_key",
        "abs_eta_gluon_fraction", "states", "mev_to_gev", "rng_user_seed",
        "mc_phi_seed_multiplier", "pseudo_data_phi_seed_multiplier", "mc_zero_seed", "pseudo_data_zero_seed"});
    rapidjson::Document payload;
    payload.Parse<rapidjson::kParseFullPrecisionFlag>(configuration.json().c_str());
    if (payload.HasParseError() || !payload.IsObject()) throw std::invalid_argument("Invalid jet-uncertainty payload");

    const auto jet_definition = configuration.string("jet_definition");
    const auto mc_type = configuration.string("mc_type");
    const auto prefix = configuration.string("group_prefix");
    energy_scale = static_cast<float>(configuration.real("mev_to_gev"));
    random = {configuration.integer("rng_user_seed"), configuration.real("mc_phi_seed_multiplier"),
              configuration.real("pseudo_data_phi_seed_multiplier"), configuration.integer("mc_zero_seed"),
              configuration.integer("pseudo_data_zero_seed")};
    if (!std::isfinite(energy_scale) || energy_scale <= 0.0f ||
        !std::isfinite(random.mc_multiplier) || random.mc_multiplier <= 0.0 ||
        !std::isfinite(random.pseudo_data_multiplier) || random.pseudo_data_multiplier <= 0.0 ||
        random.mc_fallback <= 0 || random.pseudo_data_fallback <= 0 || random.user_seed < 0) {
      throw std::invalid_argument("Invalid jet-uncertainty scale or random settings");
    }

    TEnv settings;
    const auto& entries = field(payload, "settings");
    if (!entries.IsObject()) throw std::invalid_argument("The uncertainty TEnv settings must be an object");
    for (const auto& item : entries.GetObject()) {
      if (!item.value.IsString()) throw std::invalid_argument("The uncertainty TEnv values must be strings");
      settings.SetValue(item.name.GetString(), item.value.GetString(), kEnvGlobal);
    }
    if (settings.GetValue("CombMassWeightCaloHist", "")[0] ||
        settings.GetValue("CombMassWeightTAHist", "")[0]) {
      throw std::invalid_argument("Combined-mass weight histograms are outside these recommendations");
    }

    TDirectory::TContext directory;
    TMemFile histograms("correctionlib-uncertainty-histograms.root", "RECREATE");
    TMemFile analysis("correctionlib-uncertainty-analysis.root", "RECREATE");
    write_histograms(histograms, field(payload, "histograms"));
    write_histograms(analysis, field(payload, "analysis_histograms"));
    const auto fraction = configuration.string("gluon_fraction_key");
    const auto fraction_error = configuration.string("gluon_fraction_error_key");
    const auto& abs_eta = field(payload, "abs_eta_gluon_fraction");
    if (!abs_eta.IsBool()) throw std::invalid_argument("Invalid flavour eta convention");
    const UncertaintyFlavourInput flavour{analysis, fraction, fraction_error, abs_eta.GetBool()};
    groups = initialize_uncertainty_groups(settings, histograms, flavour, jet_definition, mc_type, prefix, energy_scale);

    const TString file_validity = settings.GetValue("FileValidHistogram", "");
    if (!file_validity.IsNull()) {
      const auto parametrization = jet::CompParametrization::stringToEnum(
          settings.GetValue("FileValidHistParam", ""));
      const auto mass_definition = jet::CompMassDef::stringToEnum(settings.GetValue("FileValidHistMassDef", ""));
      auto validity_histogram = std::make_unique<jet::ValidityHistogram>(
          file_validity + "_" + jet_definition.c_str(), parametrization, energy_scale, mass_definition);
      require_success(validity_histogram->initialize(&histograms), "Could not load file-wide jet validity");
      validity = std::move(validity_histogram);
    }
    auto resolution_helper = std::make_unique<jet::ResolutionHelper>("CorrectionLibJetUncertainty", jet_definition);
    require_success(resolution_helper->initialize(settings, &histograms, mc_type.c_str()),
                    "Could not load nominal jet resolutions");
    resolutions = std::move(resolution_helper);

    std::vector<jet::UncertaintyGroup*> group_pointers;
    for (const auto& group : groups) {
      group_pointers.push_back(group.get());
      for (Scale scale : group->getScaleVars()) {
        if (!is_supported_scale(scale)) {
          throw std::invalid_argument("Unexpected substructure or other output in the configured uncertainty groups");
        }
      }
    }

    auto nominal_set = std::make_unique<jet::UncertaintySet>("NOSYS");
    require_success(nominal_set->initialize(CP::SystematicSet(), group_pointers), "Could not configure nominal uncertainty");
    nominal = std::move(nominal_set);
    const auto& configured = field(payload, "states");
    if (!configured.IsObject() || configured.ObjectEmpty()) {
      throw std::invalid_argument("No active uncertainty states in the payload");
    }
    for (const auto& state : configured.GetObject()) {
      const auto& base = field(state.value, "basename");
      const auto& parameter = field(state.value, "parameter");
      const auto& pseudo = field(state.value, "pseudo_data");
      if (!base.IsString() || !parameter.IsInt() || (parameter.GetInt() != -1 && parameter.GetInt() != 1) ||
          !pseudo.IsBool()) {
        throw std::invalid_argument("Invalid initialized jet systematic in payload");
      }
      const std::string name(state.name.GetString(), state.name.GetStringLength());
      CP::SystematicSet request;
      request.insert(CP::SystematicVariation(base.GetString(), static_cast<float>(parameter.GetInt())));
      auto set = std::make_unique<jet::UncertaintySet>(request.name());
      require_success(set->initialize(request, group_pointers), "Could not configure jet systematic " + name);
      if (!states.emplace(name, ConfiguredState{std::move(set), pseudo.GetBool()}).second) {
        throw std::invalid_argument("Duplicate initialized jet systematic " + name);
      }
    }
  }

  JetUncertaintyResult evaluate(const JetUncertaintyInput& input, const std::string& systematic) const {
    const auto found = states.find(systematic);
    const bool pseudo = found != states.end() && found->second.pseudo_data;
    const jet::UncertaintySet& state = found == states.end() ? *nominal : *found->second.uncertainty;
    JetUncertaintyResult result{input.p4, std::nullopt, std::nullopt};
    if (!pseudo) result.is_bjet = input.parton_truth_label == 5;
    if (!input.preselected) return result;

    result.status = Status::error;
    if (!input.event_available) return result;

    xAOD::Jet calibrated;
    calibrated.makePrivateStore();
    set_jet_inputs(calibrated, input, result.is_bjet);
    if (validity && !validity->getValidity(calibrated)) {
      result.status = Status::out_of_validity;
      return result;
    }

    xAOD::EventInfo event;
    event.makePrivateStore();
    event.setAverageInteractionsPerCrossing(input.average_mu);
    static const SG::Accessor<float> npv("NPV");
    npv(event) = input.uncertainty_npv;
    std::vector<std::pair<Scale, double>> uncertainties;
    const auto validity_results = state.getValidUncertaintySet(uncertainties, calibrated, event);
    if (std::any_of(validity_results.begin(), validity_results.end(),
                    [](const auto& entry) { return !entry.second; })) {
      result.status = Status::out_of_validity;
      return result;
    }

    using namespace jet::CompScaleVar;
    for (const auto& [scale, uncertainty] : uncertainties) {
      const double shift = 1.0 + uncertainty;
      switch (scale) {
        case FourVec:
          update_coordinates(calibrated, shift, shift);
          break;
        case Pt:
          update_coordinates(calibrated, shift, 1.0);
          break;
        case Mass:
          update_coordinates(calibrated, 1.0, shift);
          break;
        case PtRes:
        case PtResAbs:
          update_coordinates(calibrated, resolution_factor(*resolutions, calibrated, scale, state.getTopology(),
                                                           uncertainty, pseudo, energy_scale, random), 1.0);
          break;
        case FourVecRes:
        case FourVecResAbs: {
          const double smear = resolution_factor(*resolutions, calibrated, scale, state.getTopology(), uncertainty,
                                                 pseudo, energy_scale, random);
          update_coordinates(calibrated, smear, smear);
          break;
        }
        default:
          return result;
      }
    }
    result.p4 = {calibrated.pt(), calibrated.eta(), calibrated.phi(), calibrated.m()};
    result.status = Status::ok;
    return result;
  }

  std::vector<std::unique_ptr<jet::UncertaintyGroup>> groups;
  std::unique_ptr<const jet::ValidityHistogram> validity;
  std::unique_ptr<const jet::ResolutionHelper> resolutions;
  std::unique_ptr<const jet::UncertaintySet> nominal;
  std::unordered_map<std::string, ConfiguredState> states;
  float energy_scale{};
  ResolutionRandom random{};
};

JetUncertaintyModel::JetUncertaintyModel(const correction::PluginConfig& configuration)
    : model_(std::make_unique<Impl>(configuration)) {}
JetUncertaintyModel::~JetUncertaintyModel() = default;

JetUncertaintyResult JetUncertaintyModel::evaluate(const JetUncertaintyInput& input,
                                                    const std::string& systematic) const {
  return model_->evaluate(input, systematic);
}

}  // namespace jets
