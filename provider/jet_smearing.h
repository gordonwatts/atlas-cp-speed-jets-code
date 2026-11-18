#pragma once

#ifdef JETS_PORTABLE
#include "../portable/random.h"
#else
#include <TRandom3.h>
#endif

#include <algorithm>
#include <cmath>

#if defined(JETS_RNG_CACHE) || defined(JETS_RNG_CACHE_DIAGNOSTICS)
#include <array>
#include <cstdint>
#endif
#ifdef JETS_RNG_CACHE
#include <cfenv>
#endif

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

namespace detail {

#ifdef JETS_RNG_CACHE
#ifdef JETS_RNG_CACHE_LARGE
inline constexpr unsigned resolution_cache_index_bits = 18;
#else
inline constexpr unsigned resolution_cache_index_bits = 16;
#endif
inline constexpr unsigned resolution_cache_slots = 1u << resolution_cache_index_bits;
#else
inline constexpr unsigned resolution_cache_slots = 0;
#endif

#ifdef JETS_RNG_CACHE_DIAGNOSTICS
enum ResolutionCount {
  cache_hits, cache_misses, standard_draws, guarded_restarts, negative_restarts,
  nonfinite_restarts, scalar_seeds, scalar_draws, scalar_rejections, resolution_count_size,
};
inline thread_local std::array<std::uint64_t, resolution_count_size> resolution_counts{};
#define JETS_COUNT_RESOLUTION(name) (++::jets::detail::resolution_counts[::jets::detail::name])
#else
#define JETS_COUNT_RESOLUTION(name) ((void)0)
#endif

template <typename Generator>
double scalar_resolution_draw(Generator& generator, long long seed, double width) {
  JETS_COUNT_RESOLUTION(scalar_seeds);
  generator.SetSeed(seed);
  double factor;
  do {
    JETS_COUNT_RESOLUTION(scalar_draws);
    factor = generator.Gaus(1.0, width);
    if (factor < 0.0) JETS_COUNT_RESOLUTION(scalar_rejections);
  } while (factor < 0.0);
  return factor;
}

template <typename Generator>
double resolution_draw(Generator& generator, long long seed, double width) {
#ifdef JETS_RNG_CACHE
  if (seed > 0 && std::isfinite(width) && std::fegetround() == FE_TONEAREST) {
    struct Entry {
      long long seed;
      double normal;
    };
    thread_local std::array<Entry, resolution_cache_slots> entries{};
    const auto hash = static_cast<std::uint64_t>(seed) * 0x9e3779b97f4a7c15ULL;
    auto& entry = entries[hash >> (64 - resolution_cache_index_bits)];
    if (entry.seed == seed) {
      JETS_COUNT_RESOLUTION(cache_hits);
    } else {
      JETS_COUNT_RESOLUTION(cache_misses);
      generator.SetSeed(seed);
      JETS_COUNT_RESOLUTION(standard_draws);
      entry = {seed, generator.Gaus(0.0, 1.0)};
    }
    // ROOT's sampler returns mean + sigma * result after sampling a single
    // standard normal. The pinned builds use separate multiply and add.
    const double factor = 1.0 + width * entry.normal;
    if (std::isfinite(factor) && factor >= 0.0) return factor;
    if (factor < 0.0) {
      JETS_COUNT_RESOLUTION(negative_restarts);
    } else {
      JETS_COUNT_RESOLUTION(nonfinite_restarts);
    }
  } else {
    JETS_COUNT_RESOLUTION(guarded_restarts);
  }
  // Replay from the seed, including the first draw. A rejected variate
  // requires the complete ordered release stream, not another cached draw.
#endif
  return scalar_resolution_draw(generator, seed, width);
}

#undef JETS_COUNT_RESOLUTION

} // namespace detail

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

  // Every scalar draw restarts its private stream, including zero-width
  // draws. The optional cache only skips draws with no subsequent consumer.
#ifdef JETS_PORTABLE
  thread_local portable::Random generator;
#else
  thread_local TRandom3 generator;
#endif
  return detail::resolution_draw(generator, random.seed(raw_phi, pseudo_data), width);
}

} // namespace jets
