// Build this fixture twice with the same floating-point flags and histogram
// sources. Enable JETS_FAST_VARIABLE_FLOW in only one build, then compare
// stdout byte for byte. The main ROOT oracle remains an independent check.

#include "histogram.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using jets::portable::Axis;
using jets::portable::Histogram;
using jets::portable::Interpolation;
using jets::portable::UncertaintyLookup;

Axis variable(std::vector<double> edges) {
  const int bins = static_cast<int>(edges.size()) - 1;
  const double low = edges.front(), high = edges.back();
  return Axis(bins, low, high, std::move(edges));
}

std::vector<double> points(const Axis& axis) {
  const auto infinity = std::numeric_limits<double>::infinity();
  const auto nan = std::bit_cast<double>(std::uint64_t{0x7ff8000000000015});
  std::vector<double> values{-infinity, infinity, nan, -0.0, 0.0};
  const auto include = [&](double edge) {
    values.insert(values.end(), {std::nextafter(edge, -infinity), edge, std::nextafter(edge, infinity)});
  };
  for (int bin = 1; bin <= axis.GetNbins(); ++bin) {
    include(axis.GetBinLowEdge(bin));
    values.push_back(axis.GetBinCenter(bin));
  }
  include(axis.GetXmax());
  return values;
}

void dump_values(const std::vector<Axis>& axes) {
  std::array<std::vector<double>, 3> coordinates{{{0}, {0}, {0}}};
  std::size_t cells = 1;
  for (std::size_t i = 0; i < axes.size(); ++i) {
    cells *= axes[i].GetNbins() + 2;
    coordinates[i] = points(axes[i]);
  }
  std::vector<double> contents(cells);
  for (std::size_t i = 1; i < cells; ++i) contents[i] = 0.19 * i - 1.23;
  contents.front() = -0.0;
  const Histogram histogram("TH" + std::to_string(axes.size()) + "F", axes, contents);

  constexpr auto modes = std::array{Interpolation::Full, Interpolation::None,
                                   Interpolation::OnlyX, Interpolation::OnlyY};
  for (auto mode : modes) {
    if (axes.size() == 1 && mode == Interpolation::OnlyY) continue;
    const UncertaintyLookup lookup(std::make_unique<Histogram>(histogram), mode);
    std::cout << "case " << axes.size() << ' ' << static_cast<int>(mode) << '\n';
    for (double x : coordinates[0]) {
      for (double y : coordinates[1]) {
        for (double z : coordinates[2]) {
          try {
            double value;
            if (axes.size() == 1) value = lookup.getValue(x);
            else if (axes.size() == 2) value = lookup.getValue(x, y);
            else value = lookup.getValue(x, y, z);
            std::cout << std::hex << std::bit_cast<std::uint64_t>(value) << '\n';
          } catch (const std::out_of_range&) {
            // In particular, OnlyX/OnlyY retain the original projection
            // exception if a degenerate last bin clamps to overflow.
            std::cout << "range-error\n";
          }
        }
      }
    }
  }
}

}  // namespace

int main() {
  const Axis fixed(7, -0.39, 0.84);
  const auto irregular = variable({-4.0, -0.3, 0.1, 1.2, 6.8});
  const auto front_and_middle_duplicates = variable({-0.0, 0.0, 0.4, 0.4, 1.7});
  const auto end_duplicate = variable({-0.7, 0.1, 0.42, 0.42});
  const auto very_narrow = variable({0.2, 0.5, std::nextafter(0.5, 1.0), 2.3});
  dump_values({fixed});
  dump_values({irregular});
  dump_values({front_and_middle_duplicates});
  dump_values({very_narrow, fixed});
  dump_values({irregular, front_and_middle_duplicates});
  dump_values({irregular, fixed, end_duplicate});
}
