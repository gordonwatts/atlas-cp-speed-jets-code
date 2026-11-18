#pragma once

#include "jet_common.h"

#include <cstddef>
#include <cstdint>

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

 private:
  static thread_local const TruthContext* active_;
  const P4* jets_;
  const std::size_t* offsets_;
  std::size_t event_count_;
};

}  // namespace jets
