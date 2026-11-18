#include <AnaAlgorithm/AnaAlgorithm.h>
#include <AthContainers/Accessor.h>
#include <AthContainers/ConstAccessor.h>
#include <AthContainers/Decorator.h>
#include <AsgTools/AsgComponentFactories.h>
#include <AsgTools/ToolStore.h>
#include <PATInterfaces/ISystematicsTool.h>
#include <SystematicsHandles/SysListHandle.h>
#include <SystematicsHandles/SysReadHandle.h>
#include <xAODJet/JetContainer.h>

#include <cstdint>
#include <fstream>

// Preserve absence separately from each physical CorrectionCode decoration.
// The reference uses selectionAccept() and selectionReject(), both nonzero.
class ObserveValidity final : public EL::AnaAlgorithm
{
public:
  using EL::AnaAlgorithm::AnaAlgorithm;

  StatusCode initialize() override
  {
    ANA_CHECK(m_jets.initialize(m_systematics));
    ANA_CHECK(m_systematics.initialize());
    if (m_largeR)
    {
      std::ofstream out("initialized-tools.tsv");
      out << "tool\tlist\tvariation\n";
      for (const std::string name : {
          "seq1_JetUncertaintiesAlg_SmallR.uncertaintiesTool",
          "seq1_JetUncertaintiesAlg_SmallR.uncertaintiesToolPD",
          "seq1_JetUncertaintiesAlg_LargeR.uncertaintiesTool",
          "seq1_JetUncertaintiesAlg_LargeR.uncertaintiesToolPD",
          "seq1_JetFFSmearingAlg_LargeR.FFSmearingTool",
          "seq1_JvtEfficiencyAlg_FixedEffPt_SmallR_baselineJvt.efficiencyTool"})
      {
        const auto* tool = asg::ToolStore::get<CP::ISystematicsTool>(name);
        if (tool == nullptr)
        {
          out << name << "\tmissing\t\n";
          continue;
        }
        for (const auto& state : tool->affectingSystematics())
          out << name << "\taffecting\t" << state.name() << '\n';
        for (const auto& state : tool->recommendedSystematics())
          out << name << "\trecommended\t" << state.name() << '\n';
      }
      std::ofstream("process-status.txt") << std::ifstream("/proc/self/status").rdbuf();
      std::ofstream("process-library-maps.txt") << std::ifstream("/proc/self/maps").rdbuf();
    }
    return StatusCode::SUCCESS;
  }

  StatusCode execute(const EventContext& ctx) override
  {
    for (const auto& sys : m_systematics.systematicsVector())
    {
      const xAOD::JetContainer* jets = nullptr;
      ANA_CHECK(m_jets.retrieve(jets, sys, ctx));
      for (const xAOD::Jet* jet : *jets)
      {
        m_index(*jet) = static_cast<std::uint32_t>(jet->index());
        if (m_largeR)
          m_raw(*jet) = m_status.isAvailable(*jet) ? m_status(*jet) : 0u;
      }
    }
    return StatusCode::SUCCESS;
  }

private:
  CP::SysListHandle m_systematics{this};
  CP::SysReadHandle<xAOD::JetContainer> m_jets{
    this, "jets", "", "final large-R collections"};
  Gaudi::Property<bool> m_largeR{this, "largeR", false, "observe JMR status"};
  SG::Decorator<std::uint32_t> m_index{"refOriginalIndex"};
  SG::ConstAccessor<std::uint32_t> m_status{"outOfValidityJMR"};
  SG::Decorator<std::uint32_t> m_raw{"refJmrValidityOrAbsent"};
};

DECLARE_COMPONENT(ObserveValidity)
