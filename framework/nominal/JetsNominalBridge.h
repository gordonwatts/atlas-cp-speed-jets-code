#ifndef JETS_CORRECTION_FRAMEWORK_NOMINAL_H
#define JETS_CORRECTION_FRAMEWORK_NOMINAL_H

#include <xAODJet/Jet.h>

#include <string_view>

class JetEventInfo;

namespace jets::framework {

void validate_nominal(std::string_view jet_definition, std::string_view config,
                      std::string_view sequence, std::string_view area,
                      bool standard_properties);

void apply_small_pileup(xAOD::Jet& jet, JetEventInfo& event);
void apply_eta_jes(xAOD::Jet& jet, std::string_view definition, std::string_view starting_scale);
void apply_large_jms(xAOD::Jet& jet);
void apply_small_gsc(xAOD::Jet& jet, JetEventInfo& event);

}  // namespace jets::framework

#endif
