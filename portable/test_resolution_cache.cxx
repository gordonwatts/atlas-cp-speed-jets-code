#include "../provider/jet_smearing.h"

#include <TRandom3.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
#ifdef JETS_PORTABLE
using Candidate = jets::portable::Random;
#else
using Candidate = TRandom3;
#endif

std::size_t checked = 0;

void expect(bool condition, const char* context) {
  if (!condition) throw std::runtime_error(context);
}

void exact(double actual, double expected, const char* context) {
  expect(std::bit_cast<std::uint64_t>(actual) == std::bit_cast<std::uint64_t>(expected), context);
  ++checked;
}

double root_draw(long long seed, double width) {
  static TRandom3 generator;
  generator.SetSeed(seed);
  double factor = -1.0;
  while (factor < 0.0) factor = generator.Gaus(1., width);
  return factor;
}

void compare_seeded_draws() {
  Candidate candidate;
  constexpr auto nan = std::numeric_limits<double>::quiet_NaN();
  constexpr auto infinity = std::numeric_limits<double>::infinity();
  constexpr auto max = std::numeric_limits<double>::max();
  constexpr std::array widths{0.0, -0.0, 1.0e-12, 0.03, 0.175, 0.8, 2.9, 100.0, max, infinity, nan};
  constexpr std::array seeds{
      1LL, 12345LL, 54321LL, 45583453LL, 34545654LL, 0xffffffffLL, 0x100000000LL, 0x100000001LL, -7LL,
  };
  for (auto seed : seeds) {
    for (double width : widths) {
      for (int repeat = 0; repeat < 2; ++repeat)
        exact(jets::detail::resolution_draw(candidate, seed, width), root_draw(seed, width),
              "seeded ROOT draw or replay differs");
    }
  }

  // Sweep roughly one full columnar chunk of repeated seeds across several widths.
  // The seed source here is synthetic and independent of any PHYSLITE input.
  std::vector<long long> representative;
  representative.reserve(10000);
  std::uint32_t state = 0x0edcba98;
  for (int i = 0; i < 10000; ++i) {
    state = 1664525U * state + 1013904223U;
    representative.push_back(1 + state % 628318U);
  }
  for (double width : std::array{0.01, 0.04, 0.15, 0.23, 0.35, 0.5, 0.06, 0.12})
    for (auto seed : representative)
      exact(jets::detail::resolution_draw(candidate, seed, width), root_draw(seed, width),
            "multi-state ROOT draw differs");
}

void compare_recipe() {
  constexpr jets::ResolutionRandom source{0, 2.00e5, 1.23e5, 45583453, 34545654};
  for (bool pseudo : {false, true}) {
    const auto threshold = static_cast<float>(1.0 / (pseudo ? 1.23e5 : 2.00e5));
    const std::array phis{0.0f, -0.0f, 0.125f, -3.1415927f, threshold,
        std::nextafter(threshold, 0.0f), std::nextafter(threshold, 1.0f), -threshold};
    for (double phi : phis) {
      for (bool mc_only : {false, true}) {
        for (bool relative : {false, true}) {
          for (auto user : {0LL, 1LL, 42LL, -42LL}) {
            auto recipe = source;
            recipe.user_seed = user;
            long long expected_seed = user != 0 ? user * 1.00e5 * std::fabs(phi)
                                               : (pseudo ? 1.23e5 : 2.00e5) * std::fabs(phi);
            if (!expected_seed) expected_seed = pseudo ? 34545654 : 45583453;
            expect(recipe.seed(phi, pseudo) == expected_seed, "signed truncation or zero fallback differs");
            for (double variation : {0.0, -0.0, 0.005, -0.02, 5.0, -5.0}) {
              for (double nominal : {0.0, 0.15}) {
                const bool active = variation != 0.0 && !(pseudo && (mc_only || variation > 0.0))
                    && !(!pseudo && variation < 0.0 && !mc_only);
                std::vector<bool> requests;
                const double result = jets::resolution_factor(phi, variation, pseudo, mc_only, relative, recipe,
                    [&](bool mc) { requests.push_back(mc); return mc ? nominal * 0.5 : nominal; });
                expect(requests == (active ? std::vector<bool>{true, false} : std::vector<bool>{}),
                       "nominal MC/data ordering or applicability differs");
                const double factor = relative ? nominal : 1.0;
                const double width = std::sqrt(std::pow(nominal + std::fabs(variation) * factor, 2)
                                               - std::pow(nominal, 2));
                const double expected = !active ? 1.0 : user == 1 ? 0.0 : root_draw(expected_seed, width);
                exact(result, expected, "release resolution recipe differs");
              }
            }
          }
        }
      }
    }
  }
}

void verify_counts() {
  using namespace jets::detail;
  constexpr std::array labels{"cache_hits", "cache_misses", "standard_draws", "guarded_restarts",
      "negative_restarts", "nonfinite_restarts", "scalar_seeds", "scalar_draws", "scalar_rejections"};
  expect(labels.size() == resolution_counts.size(), "wrong counter shape");
  expect(resolution_counts[scalar_rejections] > 0, "fixture must exercise rejected ROOT draws");
  expect(resolution_counts[scalar_draws] == resolution_counts[scalar_seeds] + resolution_counts[scalar_rejections],
         "scalar draw or retry count differs");
#ifdef JETS_RNG_CACHE
  expect(resolution_counts[cache_hits] > 0 && resolution_counts[cache_misses] > 0,
         "fixture must exercise cache hits and misses");
  expect(resolution_counts[standard_draws] == resolution_counts[cache_misses],
         "cache misses must draw exactly one first standardized variate");
  expect(resolution_counts[guarded_restarts] > 0 && resolution_counts[negative_restarts] > 0 &&
             resolution_counts[nonfinite_restarts] > 0, "fixture did not cover all replay paths");
  expect(resolution_counts[scalar_seeds] == resolution_counts[guarded_restarts] +
             resolution_counts[negative_restarts] + resolution_counts[nonfinite_restarts],
         "a replay must restart with the original seed");
#else
  expect(resolution_counts[standard_draws] == 0 && resolution_counts[cache_hits] == 0,
         "independent scalar path used cache");
#endif
  for (std::size_t i = 0; i < labels.size(); ++i)
    std::cout << labels[i] << ' ' << resolution_counts[i] << '\n';
}
} // namespace

int main() {
  compare_seeded_draws();
  compare_recipe();
  verify_counts();
  std::cout << "cache_slots " << jets::detail::resolution_cache_slots << '\n';
  std::cout << "bitwise ROOT checks " << checked << '\n';
}
