#include "../provider/event_inputs.h"
#include "../provider/large_jmr.h"

#include <correction.h>
#include <correction_plugin.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using Jets = jets::TruthContext;
using Output = std::array<double, 6>;

constexpr std::array states{
    "JET_JMRUnc_Detector", "JET_JMRUnc_MCData", "JET_JMRUnc_MCcr", "JET_JMRUnc_MCfsr",
    "JET_JMRUnc_MCisr", "JET_JMRUnc_MCps", "JET_JMRUnc_MCpthard", "JET_JMRUnc_MCrecoil",
    "JET_JMRUnc_Noise_PreRec", "JET_JMRUnc_OutsideCalib", "JET_JMRUnc_Smoothing", "JET_JMRUnc_Stat",
    "JET_JMRUnc_TopologyQCD", "JET_JMRUnc_mc20FullSimvsmc20AF3_MC23_PreRec",
    "JET_JMRUnc_mc20UFOvsmc16UFO_PreRec", "JET_JMRUnc_mc20vsmc21_MC23_PreRec",
    "JET_JMRUnc_mc20vsmc23_MC23_PreRec", "JET_JMRUnc_mc23FullSimvsmc23AF3_MC23_PreRec",
};

struct TruthRows {
  std::array<std::int64_t, 4> offsets{0, 2, 3, 3};
  std::array<double, 3> pt{400e3, 400e3, 480e3};
  std::array<double, 3> eta{0.2, 0.2, 0.15};
  std::array<double, 3> phi{0.7, 0.7, -3.13};
  std::array<double, 3> mass{50e3, 80e3, 91e3};
};

void expect(bool condition, std::string_view what) {
  if (!condition) throw std::runtime_error(std::string(what));
}

class Binding {
 public:
  Binding(const TruthRows& rows, bool cache) {
    auto bind = cache ? jets_bind_jmr_truth : jets_bind_truth;
    const int rc = bind(rows.offsets.data(), rows.offsets.size(), rows.pt.data(), rows.eta.data(),
                        rows.phi.data(), rows.mass.data(), rows.pt.size());
    expect(rc == 0, jets_buffer_error());
  }
  ~Binding() { (void)jets_unbind_truth(); }
  Binding(const Binding&) = delete;
  Binding& operator=(const Binding&) = delete;
};

const jets::P4 first_reco(400e3, 0.2, 0.7, 65e3);
const jets::P4 far_reco(400e3, 0.2, 1.7, 65e3);
const jets::P4 wrap_reco(480e3, 0.15, 3.13, 105e3);
const jets::P4 zero_mass(400e3, 0.2, 0.7, 0.);

Output call(const correction::Correction& node, const jets::P4& reco, std::int64_t event,
            std::int64_t available, std::int64_t label, const std::string& state) {
  Output result{};
  node.evaluate_array({reco.Pt(), reco.Eta(), reco.Phi(), reco.M(), label, available, state, event},
                      result.data(), result.size());
  return result;
}

void same(const Output& a, const Output& b) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    expect(std::bit_cast<std::uint64_t>(a[i]) == std::bit_cast<std::uint64_t>(b[i]),
           "JMR output or status bits differ");
  }
}

template<class F>
void expect_error(F&& act, std::string_view message) {
  try {
    act();
  } catch (const std::exception& error) {
    expect(std::string_view(error.what()).find(message) != std::string_view::npos, "Wrong error precedence");
    return;
  }
  throw std::runtime_error("Expected a JMR error");
}

std::vector<Output> series(const correction::Correction& node) {
  std::vector<Output> results;
  for (const auto* base : states) {
    for (const auto* direction : {"__1up", "__1down"}) {
      const auto state = std::string(base) + direction;
      results.push_back(call(node, first_reco, 0, 1, 0, state));
      results.push_back(call(node, wrap_reco, 1, 1, 2, state));
      const auto matched_without_label = call(node, first_reco, 0, 0, 0, state);
      const auto unmatched_without_label = call(node, far_reco, 0, 0, 0, state);
      const auto empty_truth = call(node, first_reco, 2, 0, 0, state);
      const auto no_match_type = call(node, first_reco, 0, 1, 9, state);
      const auto kinematic_veto = call(node, zero_mass, 0, 0, 0, state);
      expect(matched_without_label[4] == 0 && matched_without_label[5] == 1,
             "A found match requires a truth label");
      for (const auto* veto : {&unmatched_without_label, &empty_truth, &no_match_type, &kinematic_veto}) {
        expect((*veto)[4] == 1 && (*veto)[5] == 1, "Expected out-of-validity JMR status");
        results.push_back(*veto);
      }
      results.push_back(matched_without_label);
    }
  }
  return results;
}

void match_edges() {
  const auto view = Jets::event(0);
  const auto match = [&](const auto& reco, double limit) { return Jets::nearest_jmr(reco, 0, view, limit); };
  expect(match(first_reco, 0.3) == view.data && match(first_reco, 0.3)->M() == 50e3,
         "Equal-distance candidates must keep the first mass");
  expect(!match(far_reco, 0.3), "Changed reco coordinates must miss");
  expect(match(first_reco, 0.3) == view.data, "Reordered calls changed the first tie");

  const jets::P4 shifted(400e3, 0.2, 0.8, 65e3);
  const double distance = jets::truth_distance(jets::as_cartesian(shifted), *view.data);
  expect(!match(shifted, distance), "Exact float-distance threshold must be exclusive");
  expect(match(shifted, std::nextafter(distance, std::numeric_limits<double>::infinity())) == view.data,
         "The immediately larger threshold must accept the first match");
  expect(!match(shifted, distance), "Changing the threshold must invalidate a lookup");
  expect(!Jets::nearest_jmr(first_reco, 2, Jets::event(2), 0.3), "Empty ranges have no match");

  std::array forged{jets::P4(400e3, 0.2, 0.7, 200e3)};
  const jets::TruthJetRange borrowed{forged.data(), forged.size()};
  expect(Jets::nearest_jmr(first_reco, 0, borrowed, 0.3) == borrowed.data,
         "An explicit caller range must remain scalar even during an owned lease");
  forged.front() = far_reco;
  expect(!Jets::nearest_jmr(first_reco, 0, borrowed, 0.3), "Caller-range mutation must be observable");
}
}  // namespace

int main(int argc, char** argv) {
  expect(argc == 3, "usage: test-jmr-match-cache PROVIDER PAYLOAD");
  correction::PluginRegistry registry;
  registry.load_library(argv[1]);
  const auto bundle = correction::CorrectionSet::from_file(argv[2], registry);
  const auto node = bundle->at("large_jmr");

  auto inactive = call(*node, first_reco, -1, 2, 0, "NOSYS");
  expect(std::isnan(inactive[4]) && inactive[5] == 0, "Inactive states must not touch a context");
  const auto up = std::string(states[0]) + "__1up";
  expect_error([&] { call(*node, zero_mass, -1, 2, 0, up); }, "truth label");
  expect_error([&] { call(*node, zero_mass, -1, 1, 0, up); }, "No truth-jet context");

  TruthRows rows;
  std::vector<Output> reference;
  {
    const Binding borrowed(rows, false);
    reference = series(*node);
  }
  {
    const Binding owned(rows, true);
    expect_error([&] { call(*node, zero_mass, -1, 1, 0, up); }, "event index");
    rows.mass[0] = 99e3;
    rows.phi[0] = 1.8;
    rows.offsets[1] = 1;
    match_edges();
    const auto candidate = series(*node);
    expect(candidate.size() == reference.size(), "JMR state/jet output count differs");
    for (std::size_t i = 0; i < candidate.size(); ++i) same(candidate[i], reference[i]);
  }
  expect_error([&] { call(*node, first_reco, 0, 1, 0, up); }, "No truth-jet context");

  {
    const Binding rebound(rows, true);
    const auto view = Jets::event(0);
    expect(!Jets::nearest_jmr(first_reco, 0, view, 0.3), "Rebinding must discard an old matched key");
    expect(Jets::nearest_jmr(far_reco, 0, view, 0.3) == view.data, "Rebinding must discard an old no-match key");
  }
  {
    std::array actual{first_reco};
    const std::array<std::size_t, 2> offsets{0, 1};
    const Jets borrowed(actual.data(), actual.size(), offsets.data(), offsets.size());
    const auto view = Jets::event(0);
    expect(Jets::nearest_jmr(first_reco, 0, view, 0.3) == view.data, "Borrowed scalar first match differs");
    actual.front() = far_reco;
    expect(!Jets::nearest_jmr(first_reco, 0, view, 0.3), "Borrowed scalar mutation must be observable");
  }
  std::cout << reference.size() << " exact six-value comparisons across " << 2 * states.size()
            << " FF states, plus lifetime and match-order edges\n";
}
