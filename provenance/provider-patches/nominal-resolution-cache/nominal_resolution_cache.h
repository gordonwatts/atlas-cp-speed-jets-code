#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

#if defined(__linux__) && defined(__GLIBC__) && defined(__USE_GNU) && defined(__x86_64__) && defined(__SSE2__)
#include <xmmintrin.h>
#endif

namespace jets::portable {
// The provider creates model IDs and accesses this cache from one translation
// unit. Keep their storage local to that unit and its loaded provider.
namespace {

// IDs are independent of addresses, which may be reused after a correction
// set is released. Saturate rather than ever assigning an earlier model's ID.
inline std::uint64_t nominal_resolution_identity() noexcept {
  static std::atomic<std::uint64_t> next{1};
  constexpr auto unavailable = std::numeric_limits<std::uint64_t>::max();
  auto token = next.load(std::memory_order_relaxed);
  while (token != unavailable) {
    if (next.compare_exchange_weak(token, token + 1, std::memory_order_relaxed))
      return token;
  }
  return 0;
}

inline std::optional<unsigned> nominal_resolution_flags() noexcept {
#if defined(__linux__) && defined(__GLIBC__) && defined(__USE_GNU) && defined(__x86_64__) && defined(__SSE2__)
  constexpr unsigned status_bits = 0x003f;
  constexpr unsigned denormals_are_zero = 0x0040;
  constexpr unsigned all_masks = _MM_MASK_MASK;
  constexpr unsigned required_controls =
      all_masks | denormals_are_zero | _MM_ROUND_MASK | _MM_FLUSH_ZERO_MASK;
  const auto csr = _mm_getcsr();
  if ((csr & required_controls) != all_masks || std::fegetround() != FE_TONEAREST || ::fegetexcept() != 0)
    return std::nullopt;
  return (csr & status_bits) | (static_cast<unsigned>(std::fetestexcept(FE_ALL_EXCEPT)) << 8);
#else
  return std::nullopt;
#endif
}

template<unsigned index_bits>
class NominalResolutionCache {
public:
  static_assert(index_bits > 0 && index_bits <= 20);
  static constexpr std::size_t slot_count = std::size_t{1} << index_bits;

  struct Key {
    std::uint64_t model;
    std::uint64_t pt;
    std::uint64_t abs_eta;
    bool operator==(const Key&) const = default;
  };

  struct Query {
    Key key{};
    double sigma_mc = 0;
    std::optional<double> cached_data;
    unsigned required_flags = 0;
  };

  static Key key(std::uint64_t model, double raw_pt, double raw_eta) noexcept {
    static_assert(sizeof(double) == sizeof(std::uint64_t) && std::numeric_limits<double>::is_iec559);
    constexpr auto magnitude = 0x7fffffffffffffffULL;
    return {model, std::bit_cast<std::uint64_t>(raw_pt), std::bit_cast<std::uint64_t>(raw_eta) & magnitude};
  }

  static std::size_t bucket(const Key& value) noexcept {
    // Mix both raw coordinates and the model. pT and eta usually originate
    // as float values, so large stretches of their double mantissas are zero.
    auto mixed = value.pt ^ std::rotl(value.abs_eta, 27) ^ (value.model * 0x9e3779b97f4a7c15ULL);
    mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9ULL;
    mixed = (mixed ^ (mixed >> 27)) * 0x94d049bb133111ebULL;
    return (mixed ^ (mixed >> 31)) & (slot_count - 1);
  }

  template<class Read>
  double mc(Query& query, std::uint64_t model, double raw_pt, double raw_eta, Read&& scalar) {
    query = {};
    const auto coordinates = key(model, raw_pt, raw_eta);
    constexpr auto exponent = 0x7ff0000000000000ULL;
    if (!coordinates.model || (coordinates.pt & exponent) == exponent ||
        (coordinates.abs_eta & exponent) == exponent)
      return scalar();

    const auto flags = nominal_resolution_flags();
    if (!flags) return scalar();
    const auto& entry = entries_[bucket(coordinates)];
    if (entry.key == coordinates && covers(*flags, entry.required_flags)) {
      query = {coordinates, entry.sigma_mc, entry.sigma_data, entry.required_flags};
      return entry.sigma_mc;
    }

    // Read MC here and data at its original call site. An exception or an
    // unsupported environment leaves no entry behind.
    const double result = scalar();
    if (const auto after = nominal_resolution_flags())
      query = {coordinates, result, std::nullopt, *after};
    return result;
  }

  template<class Read>
  double data(Query& pending, Read&& scalar) {
    const auto query = pending;
    pending = {};
    if (!query.key.model) return scalar();
    if (query.cached_data) {
      const auto flags = nominal_resolution_flags();
      return flags && covers(*flags, query.required_flags) ? *query.cached_data : scalar();
    }
    const double result = scalar();
    // A miss may inherit sticky flags. Requiring the entire observed set on
    // a hit is conservative and avoids clearing or raising flags in cache code.
    if (const auto flags = nominal_resolution_flags(); flags && covers(*flags, query.required_flags))
      entries_[bucket(query.key)] = {query.key, query.sigma_mc, result, *flags};
    return result;
  }

  static NominalResolutionCache& local() noexcept {
    thread_local NominalResolutionCache cache;
    return cache;
  }

private:
  struct Entry {
    Key key{};
    double sigma_mc = 0;
    double sigma_data = 0;
    unsigned required_flags = 0;
  };

  static bool covers(unsigned current_flags, unsigned required_flags) noexcept {
    return (current_flags & required_flags) == required_flags;
  }

  std::array<Entry, slot_count> entries_{};
};

using ProductionNominalResolutionCache = NominalResolutionCache<14>;

} // namespace
} // namespace jets::portable
