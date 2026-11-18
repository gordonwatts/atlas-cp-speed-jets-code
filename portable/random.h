// SPDX-License-Identifier: LGPL-2.1-or-later
// Adapted from ROOT 6.40.02 TRandom and TRandom3. See ROOT-NOTICE.md.

#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>

namespace jets::portable {

// TRandom3's positive-seed branch uses the standardized MT19937 initialization,
// twist and temper. ROOT discards zero words before returning uniform doubles.
class Random {
 public:
  explicit Random(std::uint64_t seed = 4357) { SetSeed(seed); }

  void SetSeed(std::uint64_t seed) {
    if (seed == 0) throw std::invalid_argument("A deterministic ROOT-compatible stream needs a positive seed");
    stream_.seed(static_cast<std::uint32_t>(seed));
  }

  double Rndm() {
    auto word = stream_();
    while (word == 0) word = stream_();
    return static_cast<double>(word) * 0x1p-32;
  }

  // ROOT's acceptance-complement ratio sampler. There is no cached spare
  // normal, and a zero sigma still advances the uniform stream.
  double Gaus(double mean, double sigma) { return mean + sigma * normal(); }

 private:
  double normal() {
    const double first = Rndm();
    if (first > 0.516058551) return 4.132731354 * first - 3.132731354;
    if (first < 0.107981933) {
      const double centered = 18.52161694 * first - 1.0;
      return centered > 0.0 ? 1.0 + centered : -1.0 + centered;
    }
    if (first < 0.483941449) {
      const double trial = Rndm();
      const double centered = trial - 1.0 + trial;
      const double reflected = centered > 0.0 ? 2.0 - centered : -2.0 - centered;
      if ((1.448242853 - first) * (1.46754004 + std::abs(reflected)) < 3.307147487) {
        return reflected;
      }
      const double squared = centered * centered;
      if ((first + 1.036467755) * (3.631288474 + squared) < 5.295844968) return centered;
      if (0.591923442 - first < std::exp(-(reflected * reflected + 0.4515827053) / 2.0)) {
        return reflected;
      }
      if (first + 0.375959516 < std::exp(-(squared + 0.4515827053) / 2.0)) return centered;
    }

    while (true) {
      double x = Rndm();
      double y = 0.187308492 * Rndm();
      double candidate;
      if (0.4571828819 - 0.7270572718 * x - y > 0.0) {
        candidate = 2.0 + y / x;
      } else {
        x = 1.0 - x;
        y = 0.187308492 - y;
        candidate = -(2.0 + y / x);
      }
      if ((y - 0.8853395638 + x) * (0.2770276848 + x) + 0.2452635696 < 0.0) {
        return candidate;
      }
      if (y < x + 0.03895759111 && candidate * candidate < 4.0 * (0.5029324303 - std::log(x))) {
        return candidate;
      }
    }
  }

  std::mt19937 stream_;
};

}  // namespace jets::portable
