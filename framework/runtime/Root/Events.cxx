#include "JetsCorrectionRuntime/Events.h"
#include "JetsCorrectionRuntime/EventInputsProvider.h"

#include <stdexcept>
#include <utility>

namespace {
using jets::framework::EventLease;
using jets::framework::provider_symbol;

template<class T>
T function(const char* name) {
  return reinterpret_cast<T>(provider_symbol(name));
}

void check(int status) {
  if (!status) return;
  static const auto error = function<decltype(&jets_buffer_error)>("jets_buffer_error");
  throw std::runtime_error(error());
}

auto offsets(std::size_t size) {
  return std::make_unique<EventLease::Offsets>(
      EventLease::Offsets{0, static_cast<std::int64_t>(size)});
}
}  // namespace

namespace jets::framework {

EventLease::EventLease(Release release, std::unique_ptr<Offsets> offsets) noexcept
    : release_(release), offsets_(std::move(offsets)) {}

EventLease::~EventLease() {
  (void)release_();
}

EventLease borrow_truth(const FourColumns& coordinates) {
  static const auto bind_buffer = function<decltype(&jets_bind_truth)>("jets_bind_truth");
  static const auto release_buffer = function<decltype(&jets_unbind_truth)>("jets_unbind_truth");
  const auto n = coordinates.front().size();
  for (const auto& column : coordinates)
    if (column.size() != n) throw std::invalid_argument("inconsistent framework truth coordinates");
  auto index = offsets(n);
  check(bind_buffer(index->data(), index->size(), coordinates[0].data(), coordinates[1].data(),
                    coordinates[2].data(), coordinates[3].data(), n));
  return EventLease(release_buffer, std::move(index));
}

EventLease borrow_vertices(const std::vector<std::int32_t>& types,
                           const std::vector<std::int32_t>& track_counts) {
  static const auto bind_buffer = function<decltype(&jets_bind_vertices)>("jets_bind_vertices");
  static const auto release_buffer = function<decltype(&jets_unbind_vertices)>("jets_unbind_vertices");
  if (types.size() != track_counts.size()) throw std::invalid_argument("inconsistent framework vertices");
  auto index = offsets(types.size());
  check(bind_buffer(index->data(), index->size(), types.data(), track_counts.data(), types.size()));
  return EventLease(release_buffer, std::move(index));
}

EventLease borrow_weights(const std::vector<double>& weights) {
  static const auto bind_buffer = function<decltype(&jets_bind_weights)>("jets_bind_weights");
  static const auto release_buffer = function<decltype(&jets_unbind_weights)>("jets_unbind_weights");
  auto index = offsets(weights.size());
  check(bind_buffer(index->data(), index->size(), weights.data(), weights.size()));
  return EventLease(release_buffer, std::move(index));
}

}  // namespace jets::framework
