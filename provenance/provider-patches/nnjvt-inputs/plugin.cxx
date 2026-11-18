// Built once beside each independent source snapshot in the fixture.
#include "small_jvt.cxx"

#ifdef JETS_NNJVT_REUSE_INPUTS
#include <array>
#include <optional>

namespace {
thread_local std::optional<NnJvtInputLease> hold;
constexpr std::array<double, 4> sentinels{91.5, -22.25, 0.5, -0.75};
}
#endif

extern "C" __attribute__((visibility("default"))) int correctionlib_plugin_api_version() {
  return correction::plugin_api_version;
}

extern "C" __attribute__((visibility("default"))) const char* correctionlib_plugin_build_id() {
  return correction::plugin_build_id().c_str();
}

extern "C" __attribute__((visibility("default"))) void
correctionlib_register_nodes_v4(correction::PluginRegistry& registry) {
  register_small_jvt(registry);
}

#ifdef JETS_NNJVT_REUSE_INPUTS
extern "C" __attribute__((visibility("default"))) const double* nnjvt_fixture_hold() {
  if (hold) return nullptr;
  const auto* nodes = hold.emplace().get();
  if (!nodes || nodes->size() != 1 || nodes->front().size() != 4) {
    hold.reset();
    return nullptr;
  }
  auto* writable = hold->get()->front().data();
  std::copy(sentinels.begin(), sentinels.end(), writable);
  return writable;
}

extern "C" __attribute__((visibility("default"))) bool nnjvt_fixture_release() {
  if (!hold) return false;
  const auto* nodes = hold->get();
  const bool unchanged = nodes->size() == 1 && nodes->front().size() == 4 &&
                         std::equal(sentinels.begin(), sentinels.end(), nodes->front().data());
  hold.reset();
  return unchanged;
}

extern "C" __attribute__((visibility("default"))) bool nnjvt_fixture_unwind() {
  struct Marker {};
  const double* acquired = nullptr;
  try {
    NnJvtInputLease lease;
    if (!lease.get()) return false;
    acquired = lease.get()->front().data();
    throw Marker{};
  } catch (const Marker&) {
    NnJvtInputLease after;
    return after.get() && after.get()->front().data() == acquired;
  }
}
#endif
