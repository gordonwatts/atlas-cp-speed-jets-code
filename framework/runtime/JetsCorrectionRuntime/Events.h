#ifndef JETS_CORRECTION_FRAMEWORK_EVENTS_H
#define JETS_CORRECTION_FRAMEWORK_EVENTS_H

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace jets::framework {

using FourColumns = std::array<std::vector<double>, 4>;

class EventLease {
public:
  using Offsets = std::array<std::int64_t, 2>;
  using Release = int (*)();

  EventLease(Release release, std::unique_ptr<Offsets> offsets) noexcept;
  ~EventLease();

  EventLease(const EventLease&) = delete;
  EventLease& operator=(const EventLease&) = delete;

private:
  Release release_;
  std::unique_ptr<Offsets> offsets_;
};

EventLease borrow_truth(const FourColumns& coordinates);
EventLease borrow_vertices(const std::vector<std::int32_t>& types,
                           const std::vector<std::int32_t>& track_counts);
EventLease borrow_weights(const std::vector<double>& weights);

template <class JetContainer>
FourColumns truth_columns(const JetContainer& jets) {
  FourColumns columns;
  for (auto& column : columns) column.reserve(jets.size());
  for (const auto* jet : jets) {
    columns[0].push_back(jet->pt());
    columns[1].push_back(jet->eta());
    columns[2].push_back(jet->phi());
    columns[3].push_back(jet->m());
  }
  return columns;
}

template <class VertexContainer>
std::array<double, 5> event_inputs(const VertexContainer& vertices,
                                  std::int64_t pv, bool has_pv, double npv, bool has_npv);

}  // namespace jets::framework

#include "JetsCorrectionRuntime/Events.icc"

#endif
