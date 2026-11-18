#include "jet_smearing.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

// Copies the current thread's counters when the caller provides space.
// A null pointer or smaller buffer queries the required length.
extern "C" std::size_t jets_resolution_rng_counts(std::uint64_t* output, std::size_t size) noexcept {
  const auto& counts = jets::detail::resolution_counts;
  if (output && size >= counts.size())
    std::copy(counts.begin(), counts.end(), output);
  return counts.size();
}
