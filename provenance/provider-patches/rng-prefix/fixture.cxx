#include "portable/random.h"

#include <TRandom3.h>

#include <array>
#include <bit>
#include <cfenv>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

using jets::portable::Random;
using jets::portable::random_detail::FirstMtWords;
using jets::portable::random_detail::uniform_nonzero;

namespace {

void require(bool pass, const char* context) {
  if (!pass) throw std::runtime_error(context);
}

bool exact(double first, double second) {
  return std::bit_cast<std::uint64_t>(first) == std::bit_cast<std::uint64_t>(second);
}

template<unsigned Limit>
void raw(std::uint32_t seed, unsigned words) {
  std::mt19937 reference(seed), continuation(0xa7139u);
  FirstMtWords<Limit> cursor(seed, continuation);
  for (unsigned word = 0; word < words; ++word)
    require(cursor() == reference(), "raw MT19937 word differs");
}

void known_answers() {
  constexpr std::uint32_t seed = 5489;
  const std::array<std::uint32_t, 5> expected{
    3499211612u, 581869302u, 3890346734u, 3586334585u, 545404204u
  };
  std::mt19937 continuation(0xa7139u);
  FirstMtWords<2> cursor(seed, continuation);
  for (auto word : expected)
    require(cursor() == word, "published seed-5489 sequence differs across promotion");

  const std::array<std::uint32_t, 5> scripted{0u, 0u, 1u, 0u, 0xffffffffu};
  unsigned i = 0;
  auto next = [&] { return scripted.at(i++); };
  require(exact(uniform_nonzero(next), 0x1p-32) && i == 3, "discard leading zero words");
  require(exact(uniform_nonzero(next), 0x1.fffffffe00000p-1) && i == 5, "discard zero before maximum word");
}

struct RestoreFenv {
  std::fenv_t original;
  RestoreFenv() { require(std::fegetenv(&original) == 0, "save original floating-point environment"); }
  ~RestoreFenv() { std::fesetenv(&original); }
};

void gaussian(Random& reference, Random& candidate, TRandom3& root, std::uint64_t seed, int mode) {
  RestoreFenv restore;
  reference.SetSeed(seed);
  require(std::fesetround(mode) == 0 && std::feclearexcept(FE_ALL_EXCEPT) == 0 &&
          std::feraiseexcept(FE_DIVBYZERO) == 0, "construct start environment");
  std::fenv_t start;
  require(std::fegetenv(&start) == 0, "save start environment");
  const double original = reference.Gaus(0.0, 1.0);
  const int flags = std::fetestexcept(FE_ALL_EXCEPT);
  for (bool force_single_word : {false, true}) {
    require(std::fesetenv(&start) == 0, "restore start environment");
    const double value = force_single_word ? candidate.first_standard_normal<1>(seed) :
                                             candidate.first_standard_normal(seed);
    require(exact(value, original), "first standard Gaussian differs, including forced continuation");
    require(std::fegetround() == mode && std::fetestexcept(FE_ALL_EXCEPT) == flags,
            "rounding mode or sticky flags differ");
  }
  if (mode == FE_TONEAREST) {
    root.SetSeed(seed);
    require(exact(root.Gaus(0.0, 1.0), original), "release ROOT first standard Gaussian differs");
  }
}

} // namespace

int main() {
  known_answers();
  const std::vector<std::uint64_t> special{
    1, 2, 4357, 5489, 12345, 386415, 386416, 628318, 628319, 45583453, 34545654,
    0xffffffffULL, 0x100000000ULL, 0x100000001ULL, std::numeric_limits<std::uint64_t>::max()
  };
  Random reference, candidate;
  TRandom3 root;
  for (const auto seed : special) {
    const auto narrowed = static_cast<std::uint32_t>(seed);
    raw<1>(narrowed, 1260);
    raw<16>(narrowed, 1260);
    raw<227>(narrowed, 1260);
    for (const int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO})
      gaussian(reference, candidate, root, seed, mode);
  }
  std::uint32_t bits = 0xd8d1073bu;
  for (unsigned i = 0; i < 16'384; ++i) {
    bits = 1664525u * bits + 1013904223u;
    const auto seed = std::uint64_t{1} + bits % ((i & 1u) ? 628319u : 386416u);
    raw<16>(static_cast<std::uint32_t>(seed), 23);
    gaussian(reference, candidate, root, seed, FE_TONEAREST);
  }
  try {
    candidate.first_standard_normal(0);
    throw std::runtime_error("zero seed must throw");
  } catch (const std::invalid_argument&) {
    // The released helper prohibits the nondeterministic zero-seed stream.
  }
  std::cout << "PASS known answers, zero raw words, prefix and full continuation, "
               "16,384 fit-domain seeds, boundary seeds and identical floating-point flags\n";
}
