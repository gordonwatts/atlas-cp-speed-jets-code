#include "uncertainty_components.h"

#include <AthContainers/AuxElement.h>
#include <JetUncertainties/ELogMassEtaUncertaintyComponent.h>
#include <JetUncertainties/Helpers.h>
#include <JetUncertainties/LargeRTopologyUncertaintyComponent.h>
#include <JetUncertainties/PileupUncertaintyComponent.h>
#include <JetUncertainties/PtAbsMassUncertaintyComponent.h>
#include <JetUncertainties/PtEtaUncertaintyComponent.h>
#include <JetUncertainties/PtUncertaintyComponent.h>
#include <TEnv.h>
#include <TFile.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace {

using namespace jet;

void require_success(const StatusCode& status, const std::string& message) {
  if (status.isFailure()) throw std::runtime_error(message);
}

std::unique_ptr<UncertaintyHistogram> histogram_copy(const std::unique_ptr<UncertaintyHistogram>& histogram) {
  return histogram ? std::make_unique<UncertaintyHistogram>(*histogram) : nullptr;
}

// The release FlavourUncertaintyComponent opens its analysis ROOT file by name.
// The active recommendations have one gluon-fraction pair per jet definition.
// Evaluate that source-backed case with identical UncertaintyHistogram calls
// against an already open in-memory file restored from the payload.
class EmbeddedFlavour final : public UncertaintyComponent {
 public:
  EmbeddedFlavour(const ComponentHelper& component, const jets::UncertaintyFlavourInput& input)
      : UncertaintyComponent(component, component.flavourType == FlavourComp::Composition ? 2 : 1),
        flavour_(component.flavourType),
        abs_eta_(CompParametrization::isAbsEta(component.parametrization)),
        abs_eta_fraction_(input.abs_eta),
        truth_labels_(component.LargeRJetTruthLabels),
        analysis_file_(&input.file) {
    const bool composition = flavour_ == FlavourComp::Composition;
    if (component.uncNames.size() != (composition ? 2u : 1u)) {
      throw std::invalid_argument("Wrong number of flavour response histograms");
    }
    if (!truth_labels_.empty()) truth_label_.emplace(component.LargeRJetTruthLabelName.Data());
    if (composition) {
      const auto& first = component.uncNames.at(0);
      const auto& second = component.uncNames.at(1);
      const auto light = [](const TString& name) {
        return name.Contains("light", TString::kIgnoreCase) || name.Contains("quark", TString::kIgnoreCase);
      };
      first_gluon_ = first.Contains("glu", TString::kIgnoreCase) && light(second);
      if (!first_gluon_ && !(light(first) && second.Contains("glu", TString::kIgnoreCase))) {
        throw std::invalid_argument("Missing light-quark and gluon response pair");
      }
      second_ = std::make_unique<UncertaintyHistogram>(second, m_interpolate);
    }
    if (flavour_ != FlavourComp::bJES) {
      fraction_ = std::make_unique<UncertaintyHistogram>(input.fraction_key, m_interpolate);
      fraction_error_ = std::make_unique<UncertaintyHistogram>(input.error_key, m_interpolate);
    }
  }

  EmbeddedFlavour(const EmbeddedFlavour& source)
      : UncertaintyComponent(source), flavour_(source.flavour_), abs_eta_(source.abs_eta_),
        abs_eta_fraction_(source.abs_eta_fraction_), first_gluon_(source.first_gluon_),
        truth_labels_(source.truth_labels_), truth_label_(source.truth_label_),
        analysis_file_(source.analysis_file_), second_(histogram_copy(source.second_)),
        fraction_(histogram_copy(source.fraction_)), fraction_error_(histogram_copy(source.fraction_error_)) {}

  EmbeddedFlavour* clone() const override { return new EmbeddedFlavour(*this); }

  StatusCode initialize(TFile* histogram_file) override {
    if (UncertaintyComponent::initialize(histogram_file).isFailure()) return StatusCode::FAILURE;
    if (second_ && second_->initialize(histogram_file).isFailure()) return StatusCode::FAILURE;
    if (fraction_ && (fraction_->initialize(analysis_file_).isFailure() ||
                      fraction_error_->initialize(analysis_file_).isFailure())) {
      return StatusCode::FAILURE;
    }
    analysis_file_ = nullptr;
    return StatusCode::SUCCESS;
  }

 private:
  bool getValidityImpl(const xAOD::Jet& jet, const xAOD::EventInfo&) const override {
    return !m_validHist || getValidBool(m_validHist->getValue(jet.pt() * m_energyScale, eta_coordinate(jet)));
  }

  double getUncertaintyImpl(const xAOD::Jet& jet, const xAOD::EventInfo&) const override {
    if (truth_label_) {
      if (!truth_label_->isAvailable(jet)) return JESUNC_ERROR_CODE;
      const auto label = LargeRJetTruthLabel::intToEnum((*truth_label_)(jet));
      if (label == LargeRJetTruthLabel::UNKNOWN) return JESUNC_ERROR_CODE;
      if (std::find(truth_labels_.begin(), truth_labels_.end(), label) == truth_labels_.end()) return 0.0;
    }

    static const SG::ConstAccessor<char> is_bjet("IsBjet");
    const bool bjet = is_bjet.isAvailable(jet) && is_bjet(jet) != 0;
    const double pt = jet.pt() * m_energyScale;
    const double eta = eta_coordinate(jet);
    if (flavour_ == FlavourComp::bJES) return bjet ? m_uncHist->getValue(pt, eta) : 0.0;
    if (bjet) return 0.0;

    const double fraction_eta = abs_eta_fraction_ ? std::abs(eta) : eta;
    const double fraction = fraction_->getValue(pt, fraction_eta);
    if (flavour_ == FlavourComp::Response) return m_uncHist->getValue(pt, eta) * fraction;

    const double gluon_response = (first_gluon_ ? m_uncHist : second_.get())->getValue(pt, eta);
    const double quark_response = (first_gluon_ ? second_.get() : m_uncHist)->getValue(pt, eta);
    const double sample_response = fraction * gluon_response + (1 - fraction) * quark_response;
    if (sample_response == 0.0) return JESUNC_ERROR_CODE;
    return fraction_error_->getValue(pt, fraction_eta) * std::abs(quark_response - gluon_response) /
           sample_response;
  }

  double eta_coordinate(const xAOD::Jet& jet) const {
    return abs_eta_ ? std::abs(jet.eta()) : jet.eta();
  }

  FlavourComp::TypeEnum flavour_;
  bool abs_eta_;
  bool abs_eta_fraction_;
  bool first_gluon_ = false;
  std::vector<LargeRJetTruthLabel::TypeEnum> truth_labels_;
  std::optional<SG::ConstAccessor<int>> truth_label_;
  TFile* analysis_file_;
  std::unique_ptr<UncertaintyHistogram> second_;
  std::unique_ptr<UncertaintyHistogram> fraction_;
  std::unique_ptr<UncertaintyHistogram> fraction_error_;
};

std::unique_ptr<UncertaintyComponent> make_component(
    const ComponentHelper& component, const jets::UncertaintyFlavourInput& flavour,
    float ref_npv, float ref_mu) {
  if (component.isSpecial) {
    if (component.pileupType != PileupComp::UNKNOWN &&
        component.parametrization != CompParametrization::UNKNOWN &&
        ref_npv > 0 && ref_mu > 0) {
      return std::make_unique<PileupUncertaintyComponent>(component, ref_npv, ref_mu);
    }
    if (component.flavourType == FlavourComp::Response || component.flavourType == FlavourComp::Composition ||
        component.flavourType == FlavourComp::bJES) {
      return std::make_unique<EmbeddedFlavour>(component, flavour);
    }
    if (component.name.Contains("Large", TString::kIgnoreCase) &&
        component.name.Contains("Topology", TString::kIgnoreCase) &&
        !component.LargeRJetTruthLabels.empty()) {
      return std::make_unique<LargeRTopologyUncertaintyComponent>(component);
    }
  } else {
    switch (component.parametrization) {
      case CompParametrization::Pt:
        return std::make_unique<PtUncertaintyComponent>(component);
      case CompParametrization::PtEta:
      case CompParametrization::PtAbsEta:
        return std::make_unique<PtEtaUncertaintyComponent>(component);
      case CompParametrization::PtAbsMass:
        return std::make_unique<PtAbsMassUncertaintyComponent>(component);
      case CompParametrization::eLOGmOeAbsEta:
        return std::make_unique<ELogMassEtaUncertaintyComponent>(component);
      default:
        break;
    }
  }
  throw std::invalid_argument("Unmapped uncertainty component: " + std::string(component.name.Data()));
}

}  // namespace

std::vector<std::unique_ptr<jet::UncertaintyGroup>> jets::initialize_uncertainty_groups(
    TEnv& settings, TFile& histograms, const UncertaintyFlavourInput& flavour,
    const std::string& jet_definition, const std::string& mc_type,
    const std::string& group_prefix, float energy_scale) {
  using namespace jet;
  std::vector<std::unique_ptr<UncertaintyGroup>> groups;
  std::unordered_map<int, UncertaintyGroup*> groups_by_number;
  for (std::size_t index = 0; index < 999; ++index) {
    ConfigHelper helper(Form("JESGroup.%zu.", index), mc_type.c_str(), energy_scale);
    require_success(helper.initialize(settings), "Could not read uncertainty group settings");
    if (!helper.isGroup()) continue;
    helper.enforceGroupNamePrefix(group_prefix);
    auto group = std::make_unique<UncertaintyGroup>(*helper.getGroupInfo());
    if (group->getSubgroupNum() || !groups_by_number.emplace(group->getGroupNum(), group.get()).second) {
      throw std::invalid_argument("The active recommendations require unique top-level groups");
    }
    groups.push_back(std::move(group));
  }

  const TString npv_setting = settings.GetValue("Pileup.NPVRef", "");
  const TString mu_setting = settings.GetValue("Pileup.MuRef", "");
  const float ref_npv = npv_setting.IsNull() ? -1.0f : utils::getTypeObjFromString<float>(npv_setting);
  const float ref_mu = mu_setting.IsNull() ? -1.0f : utils::getTypeObjFromString<float>(mu_setting);
  for (std::size_t index = 0; index < 999; ++index) {
    ConfigHelper helper(Form("JESComponent.%zu.", index), mc_type.c_str(), energy_scale);
    require_success(helper.initialize(settings), "Could not read uncertainty component settings");
    if (!helper.isComponent() && !helper.isCompGroup()) continue;
    helper.enforceGroupNamePrefix(group_prefix);
    helper.setComponentJetDefSuffix(jet_definition);

    const auto& component = *helper.getComponentInfo();
    UncertaintyGroup* group;
    if (helper.isCompGroup()) {
      groups.push_back(std::make_unique<UncertaintyGroup>(*helper.getGroupInfo()));
      group = groups.back().get();
    } else {
      const auto found = groups_by_number.find(component.groupNum);
      if (found == groups_by_number.end()) throw std::invalid_argument("Unresolved uncertainty group");
      group = found->second;
    }
    const auto add_component = [&](const auto& information) {
      auto value = make_component(information, flavour, ref_npv, ref_mu);
      require_success(group->addComponent(value.get()), "Could not add uncertainty component");
      value.release();  // UncertaintyGroup owns its components.
    };
    if (component.subComps.empty()) {
      add_component(component);
    } else {
      for (const auto& name : component.subComps) {
        ComponentHelper subcomponent(component);
        subcomponent.name = name;
        subcomponent.uncNames.assign(1, name);
        subcomponent.subComps.clear();
        add_component(subcomponent);
      }
    }
  }
  for (auto& group : groups) {
    require_success(group->initialize(&histograms),
                    "Could not initialize uncertainty group " + std::string(group->getName().Data()));
  }
  return groups;
}
