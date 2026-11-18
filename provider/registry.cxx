#include <correction_plugin.h>
#include "eta_jes.h"
#include "large_jmr.h"
#include "large_jms.h"
#include "small_jvt.h"
#include "small_pileup.h"
#include "small_gsc.h"
#include "event_inputs.h"
#include "jet_uncertainty.h"

extern "C" int correctionlib_plugin_api_version() {
  return correction::plugin_api_version;
}
extern "C" const char* correctionlib_plugin_build_id() {
  return correction::plugin_build_id().c_str();
}
extern "C" void correctionlib_register_nodes_v4(correction::PluginRegistry& registry) {
  register_eta_jes(registry);
  register_large_jms(registry);
  register_large_jmr(registry);
  register_small_jvt(registry);
  register_small_pileup(registry);
  register_small_gsc(registry);
  register_event_inputs(registry);
  register_jet_uncertainty(registry);
}
