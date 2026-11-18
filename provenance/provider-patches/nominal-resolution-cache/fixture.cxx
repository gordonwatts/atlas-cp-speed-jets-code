#include "histogram.h"
#include "nominal_resolution_cache.h"

#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace portable = jets::portable;
using Cache = portable::NominalResolutionCache<3>;

namespace {

struct Point {
  double pt;
  double eta;
};

void require(bool condition, std::string_view description) {
  if (!condition) throw std::runtime_error(std::string(description));
}

bool exact(double a, double b) {
  return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

portable::UncertaintyLookup table(double shift, portable::Interpolation mode) {
  std::vector<portable::Axis> axes{
    {4, 8.0, 95.0, {8.0, 15.0, 34.0, 60.0, 95.0}},
    {3, 0.0, 3.5, {0.0, 0.75, 1.8, 3.5}}
  };
  std::vector<double> cells;
  for (int i = 0; i < 30; ++i) cells.push_back(shift + 0.0007123 * (i + 1) * (i + 2));
  auto hist = std::make_unique<portable::Histogram>("TH2D", std::move(axes), std::move(cells));
  return {std::move(hist), mode};
}

struct Model {
  const std::uint64_t id = portable::nominal_resolution_identity();
  const float scale;
  const portable::UncertaintyLookup mc;
  const portable::UncertaintyLookup data;

  Model(double shift, float energy_scale, portable::Interpolation mode)
    : scale(energy_scale), mc(table(shift, mode)), data(table(shift + 1.7, mode)) {}

  double read(Point point, bool is_mc) const {
    return (is_mc ? mc : data).getValue(point.pt * scale, std::abs(point.eta));
  }
};

struct Output {
  double mc;
  double data;
  int after_mc;
  int after_data;
  std::string order;
};

Output evaluate(Cache* cache, const Model& model, Point p) {
  Output result{};
  Cache::Query pending;
  auto mc = [&] { result.order += 'M'; return model.read(p, true); };
  auto data = [&] { result.order += 'D'; return model.read(p, false); };
  result.mc = cache ? cache->mc(pending, model.id, p.pt, p.eta, mc) : mc();
  result.after_mc = std::fetestexcept(FE_ALL_EXCEPT);
  result.data = cache ? cache->data(pending, data) : data();
  result.after_data = std::fetestexcept(FE_ALL_EXCEPT);
  return result;
}

void compare(const Output& actual, const Output& expected) {
  require(exact(actual.mc, expected.mc) && exact(actual.data, expected.data), "lookup output bits");
  require(actual.after_mc == expected.after_mc && actual.after_data == expected.after_data,
          "sticky flags at both original read sites");
}

Output same_start(Cache& cache, const Model& model, Point point) {
  std::fenv_t saved;
  require(std::fegetenv(&saved) == 0, "capture test environment");
  const auto expected = evaluate(nullptr, model, point);
  require(std::fesetenv(&saved) == 0, "restore identical test environment");
  auto observed = evaluate(&cache, model, point);
  compare(observed, expected);
  return observed;
}

void coordinates_and_models() {
  Cache cache;
  Model small(0.17, 1.0f, portable::Interpolation::OnlyX);
  Model large(2.315, 0.5f, portable::Interpolation::Full);
  std::size_t points = 0;
  for (double pt : {-8.0, -0.0, 0.0, 8.0, 11.5, std::nextafter(15.0, 0.0), 15.0,
                    std::nextafter(15.0, 20.0), 22.8, 34.0, 60.0, 95.0, 188.0}) {
    for (double eta : {-4.0, -1.8, -0.0, 0.0, 0.36, 0.75, std::nextafter(0.75, 1.0), 1.8, 3.5, 4.1}) {
      for (const auto* model : {&small, &large}) {
        same_start(cache, *model, {pt, eta});
        require(same_start(cache, *model, {pt, eta}).order.empty(), "same model/point warm pair");
        ++points;
      }
    }
  }
  for (Point point : {Point{std::numeric_limits<double>::infinity(), 0.3},
                      Point{24.5, std::numeric_limits<double>::quiet_NaN()}}) {
    require(same_start(cache, small, point).order == "MD", "nonfinite coordinates bypass");
    require(same_start(cache, small, point).order == "MD", "nonfinite repeats bypass");
  }

  const Point same_coordinates{25.125, 0.6375};
  auto small_out = same_start(cache, small, same_coordinates);
  auto large_out = same_start(cache, large, same_coordinates);
  require(!exact(small_out.mc, large_out.mc) && !exact(small_out.data, large_out.data), "different table identities");
  same_start(cache, small, {same_coordinates.pt, -same_coordinates.eta});
  require(same_start(cache, small, same_coordinates).order.empty(), "absolute eta normalizes sign");

  std::optional<Model> reloaded;
  const auto& first = reloaded.emplace(0.05, 1.0f, portable::Interpolation::OnlyX);
  const auto old_identity = first.id;
  const auto old_address = static_cast<const void*>(&first);
  const auto old_result = same_start(cache, first, same_coordinates);
  require(same_start(cache, first, same_coordinates).order.empty(), "warm before replacement");
  const auto& second = reloaded.emplace(7.5, 1.0f, portable::Interpolation::OnlyX);
  require(second.id != old_identity && static_cast<const void*>(&second) == old_address, "same address and new lifetime");
  const auto new_result = same_start(cache, second, same_coordinates);
  require(new_result.order == "MD" && !exact(old_result.mc, new_result.mc) && !exact(old_result.data, new_result.data),
          "replacement cannot reuse an earlier object's cached table values");
  require(same_start(cache, second, same_coordinates).order.empty(), "replacement warms independently");

  // Deliberately collide distinct x keys and show exact rejection/eviction.
  const auto key = Cache::key(small.id, same_coordinates.pt, same_coordinates.eta);
  Point collision{-991.0, same_coordinates.eta};
  while (Cache::bucket(Cache::key(small.id, collision.pt, collision.eta)) != Cache::bucket(key))
    collision.pt += 0.5;
  require(collision.pt != same_coordinates.pt, "distinct coordinates collide");
  same_start(cache, small, same_coordinates);
  require(same_start(cache, small, same_coordinates).order.empty(), "warm one cache slot");
  require(same_start(cache, small, collision).order == "MD", "collision rejected");
  require(same_start(cache, small, same_coordinates).order == "MD", "collision evicted original");
  std::cout << "coordinate/model pairs checked " << points << ", deliberate collision bucket " << Cache::bucket(key) << '\n';
}

void fenv_and_errors() {
  Cache cache;
  Model model(0.125, 0.47f, portable::Interpolation::Full);
  const Point p{48.9379, 0.6853};
  require(std::feclearexcept(FE_ALL_EXCEPT) == 0, "clear flags");
  require(same_start(cache, model, p).order == "MD", "cold FENV case");
  require(same_start(cache, model, p).order.empty(), "conventional FENV warm case");
  require(std::feclearexcept(FE_ALL_EXCEPT) == 0, "clear after warm lookup");
  auto ret = same_start(cache, model, p);
  require(ret.after_data & FE_INEXACT, "fixture really exercises inexact sticky flag");
  require(ret.order == "MD", "cleared flags trigger original reads");
  require(same_start(cache, model, p).order.empty(), "cache remains available after restoring flags");

  for (int mode : {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    require(std::fesetround(mode) == 0, "activate alternate rounding");
    require(!portable::nominal_resolution_flags(), "disable cache in nondefault rounding");
    require(same_start(cache, model, p).order == "MD", "preserve alternate lookup rounding");
  }
  require(std::fesetround(FE_TONEAREST) == 0, "restore nearest rounding");

#if defined(__linux__) && defined(__GLIBC__) && defined(__USE_GNU) && defined(__x86_64__) && defined(__SSE2__)
  const auto initial = _mm_getcsr();
  _mm_setcsr(initial | _MM_FLUSH_ZERO_MASK);
  require(same_start(cache, model, p).order == "MD", "flush-to-zero mode bypasses");
  _mm_setcsr(initial | 0x0040);
  require(same_start(cache, model, p).order == "MD", "denormals-are-zero mode bypasses");
  _mm_setcsr(initial);
  std::fenv_t environment;
  require(std::fegetenv(&environment) == 0, "snapshot trap state");
  require(::feenableexcept(FE_DIVBYZERO) >= 0, "enable an FP trap");
  require(same_start(cache, model, p).order == "MD", "unmasked FP trap bypasses");
  require(std::fesetenv(&environment) == 0, "restore trap state");
#endif

  Cache::Query query;
  std::string order;
  auto read_mc = [&] { order += 'M'; return 0.45; };
  auto read_data = [&] { order += 'D'; return 1.5; };
  const auto synthetic = portable::nominal_resolution_identity();
  const auto fail_mc = [&] () -> double { order += 'M'; throw std::runtime_error("mc fail"); };
  const auto fail_data = [&] () -> double { order += 'D'; throw std::runtime_error("data fail"); };
  try {
    cache.mc(query, synthetic, p.pt, p.eta, fail_mc);
    cache.data(query, read_data);
    require(false, "MC must throw");
  } catch (const std::runtime_error& error) {
    require(error.what() == std::string_view("mc fail") && order == "M", "MC failure remains first");
  }
  order.clear();
  try {
    cache.mc(query, synthetic, p.pt, p.eta, read_mc);
    cache.data(query, fail_data);
    require(false, "data must throw");
  } catch (const std::runtime_error& error) {
    require(error.what() == std::string_view("data fail") && order == "MD", "data failure after MC");
  }
  order.clear();
  require(exact(cache.mc(query, synthetic, p.pt, p.eta, read_mc), 0.45), "successful MC after exception");
  require(exact(cache.data(query, read_data), 1.5), "successful data after exception");
  require(order == "MD", "failed pair was never published");
}

} // namespace

int main() try {
  require(portable::nominal_resolution_flags().has_value(), "fixture needs reference Linux x86_64 floating point mode");
  coordinates_and_models();
  fenv_and_errors();
  using Store = portable::ProductionNominalResolutionCache;
  std::cout << "exact cache fixture passed with " << Store::slot_count << " slots and " << sizeof(Store)
            << " bytes per active worker thread\n";
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
