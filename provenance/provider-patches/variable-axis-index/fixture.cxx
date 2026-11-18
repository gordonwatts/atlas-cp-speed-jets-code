#include "histogram.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using jets::portable::Axis;
constexpr double infinity = std::numeric_limits<double>::infinity();

int oracle(const std::vector<double>& edges, double low, double high, double x) {
  if (x < low) return 0;
  if (!(x < high)) return static_cast<int>(edges.size());
  const auto edge = std::lower_bound(edges.begin(), edges.end(), x);
  return static_cast<int>(edge - edges.begin()) + (edge != edges.end() && *edge == x);
}

std::size_t check_axis(const std::string& name, const Axis& axis,
                       const std::vector<double>& edges, double low, double high) {
  std::size_t checks = 0;
  auto check = [&](double value) {
    const int expected = oracle(edges, low, high, value);
    if (axis.FindFixBin(value) != expected || axis.legacy_find_bin(value) != expected) {
      std::cerr << name << " " << std::hexfloat << value << " expected " << expected << "\n";
      throw std::logic_error("variable bin mismatch");
    }
    checks += 2;
  };
  auto check_seam = [&](double value) {
    check(std::nextafter(value, -infinity));
    check(value);
    check(std::nextafter(value, infinity));
  };
  for (const double edge : edges) check_seam(edge);
  for (std::size_t bin = 0; bin + 1 < edges.size(); ++bin)
    check(edges[bin] + (edges[bin + 1] - edges[bin]) * 0.5);
  // Each boundary is tested at the exact split and at its adjacent ULPs.
  for (int step = 0; step <= 64; ++step)
    check_seam(low + ((high - low) / 64) * step);
  for (const double special : {-0.0, +0.0, infinity, -infinity,
       std::bit_cast<double>(std::uint64_t{0x7ff8000000000015}),
       std::bit_cast<double>(std::uint64_t{0xfff8000000000015})}) check(special);
  return checks;
}

std::size_t check(const std::string& name, const std::vector<double>& edges,
                  double low, double high) {
  const Axis axis(static_cast<int>(edges.size()) - 1, low, high, edges);
  return check_axis(name, axis, edges, low, high);
}

}  // namespace

int main() {
  std::string name;
  std::size_t checks = 0, archived = 0;
  while (std::getline(std::cin, name)) {
    std::string line, field;
    if (!std::getline(std::cin, line)) throw std::invalid_argument("missing axis geometry");
    std::istringstream fields(line);
    std::vector<double> numbers;
    while (fields >> field) {
      char* end = nullptr;
      const double value = std::strtod(field.c_str(), &end);
      if (*end) throw std::invalid_argument("invalid axis edge");
      numbers.push_back(value);
    }
    if (numbers.size() < 4) throw std::invalid_argument("short axis geometry");
    checks += check(name, {numbers.begin() + 2, numbers.end()}, numbers[0], numbers[1]);
    ++archived;
  }
  if (archived < 54) throw std::invalid_argument("incomplete archived axis geometry");

  std::vector<double> repeated;
  for (auto i = 0; i <= 64; ++i) repeated.push_back(i);
  repeated.front() = -0.0;
  repeated[1] = +0.0;
  repeated[31] = repeated[32] = 30.0;
  repeated[63] = repeated.back();
  checks += check("duplicate underflow, interior, overflow", repeated, +0.0, repeated.back());
  // ROOT projections copy the selected edge subrange, including equal edges.
  const Axis cropped(64, -0.0, repeated.back(), repeated, std::array<int, 2>{3, 61});
  const std::vector<double> subrange(repeated.begin() + 2, repeated.begin() + 62);
  checks += check_axis("restricted projection", cropped.projection_axis(), subrange,
                       subrange.front(), subrange.back());

  std::vector<double> crossing;
  for (int i = -32; i <= 32; ++i) crossing.push_back(i / 32.);
  // The tiny negative edge maps arithmetically to the zero split, but its
  // exact equality with the omitted previous edge requires the full search.
  crossing.insert(crossing.begin() + 32, -std::numeric_limits<double>::denorm_min());
  checks += check("rounded split with equal edge to the left", crossing, -1., 1.);

  std::vector<double> adjacent;
  double value = 1.;
  for (int i = 0; i <= 64; ++i) {
    adjacent.push_back(value);
    value = std::nextafter(value, infinity);
  }
  checks += check("adjacent binary64 edges", adjacent, adjacent.front(), adjacent.back());

  std::vector<double> subnormal;
  for (int i = 0; i <= 64; ++i) subnormal.push_back(i * std::numeric_limits<double>::denorm_min());
  checks += check("non-finite bucket scale fallback", subnormal, subnormal.front(), subnormal.back());
  std::cout << "Exact variable-bin comparisons " << checks << " across " << archived
            << " archived geometries and synthetic boundaries\n";
}
