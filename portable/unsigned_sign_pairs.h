#pragma once

#include "../provider/jet_uncertainty.h"

#include <atomic>
#include <bit>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <new>
#include <optional>
#include <stdexcept>
#include <vector>

namespace jets::unsigned_pairs {

inline constexpr std::size_t maximum_rows = 65536;
inline std::atomic<std::uint64_t> model_counter{0};

struct StateKey {
  std::uint64_t model;
  std::size_t group;
  int sign;
  bool pseudo;

  bool operator==(const StateKey&) const = default;

  bool opposite(const StateKey& other) const {
    return model == other.model && group == other.group && pseudo == other.pseudo && sign == -other.sign;
  }
};

inline bool same_p4(const std::array<double, 4>& a, const std::array<double, 4>& b) {
  using Bits = std::array<std::uint64_t, 4>;
  return std::bit_cast<Bits>(a) == std::bit_cast<Bits>(b);
}

inline bool same_inputs(const JetUncertaintyInput& a, const JetUncertaintyInput& b) {
  return same_p4(a.p4, b.p4) &&
         std::bit_cast<std::uint32_t>(a.average_mu) == std::bit_cast<std::uint32_t>(b.average_mu) &&
         std::bit_cast<std::uint32_t>(a.uncertainty_npv) == std::bit_cast<std::uint32_t>(b.uncertainty_npv) &&
         a.event_available == b.event_available && a.preselected == b.preselected &&
         a.parton_truth_label == b.parton_truth_label && a.large_r_truth_label == b.large_r_truth_label &&
         a.calo_scale.has_value() == b.calo_scale.has_value() &&
         (!a.calo_scale || same_p4(*a.calo_scale, *b.calo_scale));
}

struct Row {
  JetUncertaintyInput input;
  double aggregate;
};

enum class Phase { idle, pending, capture, reuse, fallback };

class Chunk {
 public:
  explicit Chunk(std::size_t rows) : rows_(rows <= maximum_rows ? rows : 0) {}

  void begin_state() {
    if (phase_ != Phase::idle) throw std::logic_error("Unsigned pair state is already active");
    phase_ = Phase::pending;
    visited_ = 0;
    current_.reset();
  }

  void end_state(bool completed) {
    if (phase_ == Phase::idle) throw std::logic_error("No unsigned pair state is active");
    if (completed && phase_ == Phase::capture && visited_ == rows_) {
      previous_ = current_;
    } else {
      previous_.reset();
    }
    phase_ = Phase::idle;
  }

  void discard() {
    if (phase_ != Phase::idle) throw std::logic_error("Cannot discard an active unsigned pair state");
    previous_.reset();
  }

  std::optional<std::size_t> visit(std::optional<StateKey> key) {
    if (phase_ == Phase::idle) {
      previous_.reset();
      return std::nullopt;
    }
    if (phase_ == Phase::fallback) return std::nullopt;
    if (!key || visited_ >= rows_ || std::fegetround() != FE_TONEAREST) {
      phase_ = Phase::fallback;
      return std::nullopt;
    }
    if (phase_ == Phase::pending) {
      current_ = key;
      phase_ = previous_ && key->opposite(*previous_) ? Phase::reuse : Phase::capture;
      if (phase_ == Phase::capture && records_.empty()) {
        try {
          records_.resize(rows_);
        } catch (const std::bad_alloc&) {
          phase_ = Phase::fallback;
          return std::nullopt;
        }
      }
    }
    if (current_ != key) {
      phase_ = Phase::fallback;
      return std::nullopt;
    }
    const auto row = visited_++;
    if (phase_ == Phase::capture) records_[row].reset();
    return row;
  }

  std::optional<double> cached(std::size_t row, const JetUncertaintyInput& input) {
    const auto& stored = records_[row];
    if (phase_ == Phase::reuse && stored && same_inputs(input, stored->input)) {
      ++hits_;
      return stored->aggregate;
    }
    return std::nullopt;
  }

  void record(std::size_t row, const JetUncertaintyInput& input, double value) {
    if (phase_ == Phase::capture) records_[row] = Row{input, value};
  }

  bool idle() const noexcept { return phase_ == Phase::idle; }
  std::size_t hits() const noexcept { return hits_; }

 private:
  std::size_t rows_;
  std::size_t visited_ = 0;
  std::size_t hits_ = 0;
  Phase phase_ = Phase::idle;
  std::optional<StateKey> previous_;
  std::optional<StateKey> current_;
  std::vector<std::optional<Row>> records_;
};

inline thread_local std::optional<Chunk> chunk;

} // namespace jets::unsigned_pairs

extern "C" {
int jets_begin_unsigned_pairs(std::size_t rows);
int jets_end_unsigned_pairs();
int jets_unsigned_pair_begin_state();
int jets_unsigned_pair_end_state(int completed);
int jets_unsigned_pair_discard();
std::size_t jets_unsigned_pair_hits();
const char* jets_unsigned_pair_error();
}
