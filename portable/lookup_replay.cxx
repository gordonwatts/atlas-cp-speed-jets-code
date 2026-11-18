#include "histogram.h"

#include <JetUncertainties/UncertaintyHistogram.h>
#include <TError.h>
#include <TFile.h>
#include <TH1.h>
#include <TKey.h>
#include <rapidjson/filereadstream.h>
#include <sched.h>
#include <time.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace portable = jets::portable;

namespace {

struct Point {
  double pt_gev;
  double eta;
};

std::vector<Point> read_points(const char* path, float scale) {
  static_assert(std::endian::native == std::endian::little);
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
  constexpr std::size_t bytes_per_point = 2 * sizeof(float);
  const auto size = std::filesystem::file_size(path);
  if (size == 0 || size % bytes_per_point != 0)
    throw std::runtime_error("Invalid float32 coordinate file");
  std::vector<Point> result;
  result.reserve(size / bytes_per_point);
  std::ifstream input(path, std::ios::binary);
  for (std::size_t i = 0; i < size / bytes_per_point; ++i) {
    std::array<float, 2> raw;
    if (!input.read(reinterpret_cast<char*>(raw.data()), bytes_per_point) ||
        !std::isfinite(raw[0]) || !std::isfinite(raw[1]))
      throw std::runtime_error("Invalid pT or eta in coordinate file");
    result.push_back({static_cast<double>(raw[0]) * scale, raw[1]});
  }
  return result;
}

auto open_portable(const char* path) {
  std::unique_ptr<std::FILE, decltype(&std::fclose)> file(std::fopen(path, "rb"), &std::fclose);
  if (!file) throw std::runtime_error("Cannot open exported table");
  std::array<char, 65536> buffer;
  rapidjson::FileReadStream input(file.get(), buffer.data(), buffer.size());
  rapidjson::Document json;
  if (json.ParseStream<rapidjson::kParseFullPrecisionFlag>(input).HasParseError())
    throw std::runtime_error("Cannot parse exported table");
  return portable::UncertaintyLookup(portable::restore_histogram(json), portable::Interpolation::Full);
}

void check_cpu() {
  cpu_set_t mask;
  if (sched_getaffinity(0, sizeof(mask), &mask) || CPU_COUNT(&mask) != 1 || !CPU_ISSET(3, &mask))
    throw std::runtime_error("Replay requires CPU3");
  auto tasks = std::filesystem::directory_iterator("/proc/self/task");
  if (std::distance(tasks, std::filesystem::directory_iterator{}) != 1)
    throw std::runtime_error("Replay requires one thread");
}

void require_equal(double authority, double candidate, std::size_t index, const Point& point) {
  if (std::bit_cast<std::uint64_t>(authority) == std::bit_cast<std::uint64_t>(candidate))
    return;
  std::cerr << std::setprecision(17) << "Point " << index << ": pT " << point.pt_gev
            << " eta " << point.eta << " release " << authority << " portable " << candidate << '\n';
  throw std::runtime_error("Nonidentical lookup result");
}

double cpu_seconds() {
  timespec stamp;
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &stamp) != 0)
    throw std::runtime_error("Could not read thread CPU time");
  return stamp.tv_sec + 1e-9 * stamp.tv_nsec;
}

struct Measurement {
  double wall_seconds;
  double cpu_seconds;
  double checksum;
};

template<class Lookup>
Measurement measure(const Lookup& lookup, const std::vector<Point>& input, int passes) {
  const double cpu_start = cpu_seconds();
  const auto wall_start = std::chrono::steady_clock::now();
  double checksum = 0;
  for (int pass = 0; pass < passes; ++pass) {
    for (const auto& point : input)
      checksum += lookup.getValue(point.pt_gev, point.eta);
  }
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
  const double cpu = cpu_seconds() - cpu_start;
  return {wall, cpu, checksum};
}

void emit(std::string_view implementation, int round, std::size_t calls, const Measurement& outcome) {
  std::cout << "sample\t" << implementation << '\t' << round << '\t' << calls << '\t'
            << outcome.wall_seconds << '\t' << outcome.cpu_seconds << '\t' << outcome.checksum << '\n';
}

}  // namespace

int main(int argc, char* argv[]) try {
  if (argc != 9)
    throw std::runtime_error("Expected ROOT file, key, key cycle, table JSON, coordinates, energy scale, passes and rounds");
  check_cpu();
  TH1::AddDirectory(false);
  gErrorIgnoreLevel = kError;
  const int expected_cycle = std::stoi(argv[3]);
  const float scale = std::stof(argv[6]);
  const int passes = std::stoi(argv[7]);
  const int rounds = std::stoi(argv[8]);
  if (!std::isfinite(scale) || scale <= 0 || passes < 1 || rounds < 2)
    throw std::runtime_error("Invalid replay scale or repetition count");
  TFile source(argv[1], "READ");
  if (source.IsZombie()) throw std::runtime_error("Could not open the release ROOT table file");
  const auto* key = source.GetKey(argv[2]);
  if (!key || key->GetCycle() != expected_cycle)
    throw std::runtime_error("Release ROOT table or recorded key cycle is missing");
  jet::UncertaintyHistogram stock(argv[2], jet::Interpolate::Full);
  if (!stock.initialize(&source).isSuccess())
    throw std::runtime_error("Could not initialize the release UncertaintyHistogram");
  const auto candidate = open_portable(argv[4]);
  if (candidate.histogram().GetDimension() != 2 ||
      std::string_view(key->GetClassName()) != candidate.histogram().ClassName())
    throw std::runtime_error("Lookup requires two matching ROOT histogram axes");
  const auto input = read_points(argv[5], scale);
  std::cout << std::setprecision(17);
  std::size_t nonzero = 0;
  auto minimum = std::numeric_limits<double>::infinity();
  auto maximum = -minimum;
  for (std::size_t i = 0; i < input.size(); ++i) {
    const auto& point = input[i];
    const double expected = stock.getValue(point.pt_gev, point.eta);
    const double got = candidate.getValue(point.pt_gev, point.eta);
    if (!std::isfinite(expected)) throw std::runtime_error("Release produced a nonfinite lookup");
    require_equal(expected, got, i, point);
    nonzero += expected != 0;
    minimum = std::min(minimum, expected);
    maximum = std::max(maximum, expected);
  }
  if (nonzero == 0 || minimum == maximum)
    throw std::runtime_error("Replay failed to exercise varied table results");
  std::cout << "proof\t" << input.size() << '\t' << nonzero << '\t' << minimum << '\t' << maximum << '\t'
            << key->GetClassName() << '\t' << key->GetCycle() << '\t' << scale << '\n';

  const auto calls = input.size() * passes;
  for (int round = 0; round < rounds; ++round) {
    const bool root_first = round % 2 == 0;
    const Measurement first = root_first ? measure(stock, input, passes) : measure(candidate, input, passes);
    const Measurement second = root_first ? measure(candidate, input, passes) : measure(stock, input, passes);
    require_equal(first.checksum, second.checksum, 0, input[0]);
    emit(root_first ? "release" : "portable", round, calls, first);
    emit(root_first ? "portable" : "release", round, calls, second);
  }
  check_cpu();
} catch (const std::exception& exception) {
  std::cerr << exception.what() << '\n';
  return 1;
}
