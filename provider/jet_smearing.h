#pragma once

#ifdef JETS_PORTABLE
#include "../portable/random.h"
#else
#include <TRandom3.h>
#endif

#include <algorithm>
#include <cmath>

namespace jets {

struct ResolutionRandom {
  long long user_seed;
  double mc_multiplier;
  double pseudo_data_multiplier;
  long long mc_fallback;
  long long pseudo_data_fallback;

  long long seed(double raw_phi, bool pseudo_data) const {
    // In the release the phi comes from the jet's float auxiliary accessor.
    // Integer conversion truncates toward zero before the fallback test.
    long long value = user_seed != 0
        ? user_seed * 1.00e+5 * std::abs(raw_phi)
        : (pseudo_data ? pseudo_data_multiplier : mc_multiplier) * std::abs(raw_phi);
    return value == 0 ? (pseudo_data ? pseudo_data_fallback : mc_fallback) : value;
  }
};

// AnalysisBase 25.2.117 JetUncertaintiesTool::getSmearingFactor. Nominal
// resolution is read for MC (true) and data (false), in this order.
template <typename ReadNominal>
double resolution_factor(double raw_phi, double variation, bool pseudo_data,
                         bool smear_only_mc, bool relative,
                         const ResolutionRandom& random, ReadNominal&& read_nominal) {
  if (variation == 0.0 || (pseudo_data && (smear_only_mc || variation > 0.0)) ||
      (!pseudo_data && variation < 0.0 && !smear_only_mc)) {
    return 1.0;
  }
  const double sigma_mc = read_nominal(true);
  const double sigma_data = read_nominal(false);
  const double sigma = std::max(sigma_mc, sigma_data);
  const double relative_factor = relative ? sigma : 1.0;
  const double width = std::sqrt(std::pow(sigma + std::abs(variation) * relative_factor, 2) -
                                 std::pow(sigma, 2));
  if (random.user_seed == 1) return 0.0;

  // Every eligible call restarts its stream before drawing, including
  // zero-width draws. Generator storage can therefore be reused.
#ifdef JETS_PORTABLE
  thread_local portable::Random generator;
#else
  thread_local TRandom3 generator;
#endif
  generator.SetSeed(random.seed(raw_phi, pseudo_data));
  double factor;
  do {
    factor = generator.Gaus(1.0, width);
  } while (factor < 0.0);
  return factor;
}

} // namespace jets
