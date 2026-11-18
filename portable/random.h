// SPDX-License-Identifier: LGPL-2.1-or-later
// Adapted from ROOT 6.40.02 TRandom and TRandom3. See ROOT-NOTICE.md.

#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>

namespace jets::portable {

namespace random_detail {

template<class NextRaw>
double uniform_nonzero(NextRaw&& next) {
  auto word = next();
  while (word == 0) word = next();
  return static_cast<double>(word) * 0x1p-32;
}

#ifdef JETS_RNG_PREFIX
// For i < 624 - 397, the first MT19937 twist reads only original
// initialized words s[i], s[i+1], and s[i+397]. Two seed cursors
// avoid building the state array and produce the requested words in order.
template<unsigned Limit>
class FirstMtWords {
  static constexpr unsigned shift = 397;
  static_assert(0 < Limit && Limit <= 624 - shift);

 public:
  FirstMtWords(std::uint32_t seed, std::mt19937& continuation)
      : seed_(seed), low_(seed), high_(seed), continuation_(continuation) {
    for (unsigned i = 1; i <= shift; ++i) high_ = seed_step(high_, i);
  }

  std::uint32_t operator()() {
    if (position_ == Limit) {
      if (!promoted_) {
        continuation_.seed(seed_);
        continuation_.discard(Limit);
        promoted_ = true;
      }
      return static_cast<std::uint32_t>(continuation_());
    }

    if (position_ != 0) high_ = seed_step(high_, shift + position_);
    const auto next_low = seed_step(low_, position_ + 1);
    const auto joined = (low_ & 0x80000000u) | (next_low & 0x7fffffffu);
    auto word = high_ ^ (joined >> 1) ^ ((joined & 1u) ? 0x9908b0dfu : 0u);
    low_ = next_low;
    ++position_;

    word ^= word >> 11;
    word ^= (word << 7) & 0x9d2c5680u;
    word ^= (word << 15) & 0xefc60000u;
    word ^= word >> 18;
    return word;
  }

 private:
  static std::uint32_t seed_step(std::uint32_t word, unsigned position) {
    return 1812433253u * (word ^ (word >> 30)) + position;
  }

  const std::uint32_t seed_;
  std::uint32_t low_;
  std::uint32_t high_;
  std::mt19937& continuation_;
  unsigned position_ = 0;
  bool promoted_ = false;
};
#endif

}  // namespace random_detail

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
    return random_detail::uniform_nonzero([this] { return stream_(); });
  }

  // ROOT's acceptance-complement ratio sampler. There is no cached spare
  // normal, and a zero sigma still advances the uniform stream.
  double Gaus(double mean, double sigma) {
    return mean + sigma * normal([this] { return Rndm(); });
  }

#ifdef JETS_RNG_PREFIX
  // One-shot result for a cache miss. A scalar caller must reseed as usual;
  // this function promises the variate, not the post-draw generator state.
  template<unsigned Limit = 16>
  double first_standard_normal(std::uint64_t seed) {
    if (seed == 0) throw std::invalid_argument("A deterministic ROOT-compatible stream needs a positive seed");
    random_detail::FirstMtWords<Limit> prefix(static_cast<std::uint32_t>(seed), stream_);
    return 0.0 + 1.0 * normal([&prefix] { return random_detail::uniform_nonzero(prefix); });
  }
#endif

 private:
  template<class Uniform>
  static double normal(Uniform&& uniform) {
    const double first = uniform();
    if (first > 0.516058551) return 4.132731354 * first - 3.132731354;
    if (first < 0.107981933) {
      const double centered = 18.52161694 * first - 1.0;
      return centered > 0.0 ? 1.0 + centered : -1.0 + centered;
    }
    if (first < 0.483941449) {
      const double trial = uniform();
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
      double x = uniform();
      double y = 0.187308492 * uniform();
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
