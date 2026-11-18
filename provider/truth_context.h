#pragma once

#include "jet_common.h"

#include <cstddef>
#include <cstdint>

#ifdef JETS_LARGE_JMR_MATCH_CACHE
#include <memory>
#endif

namespace jets {

struct TruthJetRange {
  const P4* data;
  std::size_t size;
};

// Borrow contiguous event ranges until the enclosing evaluation call returns.
// Bind, evaluate, and destroy on the same thread. The buffers remain owned by
// the caller and must outlive the binding.
class TruthContext {
 public:
  TruthContext(const P4* jets, std::size_t jet_count,
               const std::size_t* offsets, std::size_t offset_count);
  ~TruthContext();
  TruthContext(const TruthContext&) = delete;
  TruthContext& operator=(const TruthContext&) = delete;

  static TruthJetRange event(std::int64_t event_index);

#ifdef JETS_LARGE_JMR_MATCH_CACHE
  // Copy the caller's arrays into a private snapshot for a C R10 chunk.
  static std::unique_ptr<TruthContext> own_jmr(
      const std::int64_t* offsets, std::size_t offset_count, const double* pt,
      const double* eta, const double* phi, const double* mass, std::size_t rows);
  static const P4* nearest_jmr(const P4& reco, std::int64_t event_index,
                               TruthJetRange truth, double maximum_delta_r);
#endif

 private:
  static thread_local const TruthContext* active_;
  const P4* jets_;
  const std::size_t* offsets_;
  std::size_t event_count_;
#ifdef JETS_LARGE_JMR_MATCH_CACHE
  struct OwnedTruth;
  struct MatchCache;
  std::unique_ptr<const OwnedTruth> owned_;
  mutable std::unique_ptr<MatchCache> matches_;
#endif
};

}  // namespace jets
