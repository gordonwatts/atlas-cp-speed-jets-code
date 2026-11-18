#include <correction.h>
#include <correction_plugin.h>
#include <rapidjson/document.h>

#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <dlfcn.h>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <typeinfo>
#include <vector>

namespace {

using Values = std::vector<correction::Variable::Type>;
using Results = std::array<double, 4>;
constexpr Results sentinels{921.25, -49.75, 81.125, 43.9875};

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

template<class T>
T symbol(void* library, const char* name) {
  auto* address = dlsym(library, name);
  require(address != nullptr, name);
  return reinterpret_cast<T>(address);
}

std::string read(const char* path) {
  std::ifstream stream(path);
  require(stream.is_open(), path);
  std::ostringstream result;
  result << stream.rdbuf();
  require(!stream.bad() && !result.str().empty(), path);
  return result.str();
}

struct RestoreFenv {
  fenv_t state;
  RestoreFenv() { require(std::fegetenv(&state) == 0, "save caller floating-point environment"); }
  ~RestoreFenv() { std::fesetenv(&state); }
};

struct Observation {
  Results out = sentinels;
  std::string exception_class;
  std::string exception_text;
  int sticky_flags;
  int rounding;
};

template<class Node>
Observation evaluate(const Node& node, const Values& values, const fenv_t& start, size_t size) {
  require(std::fesetenv(&start) == 0, "restore the same start environment for each arm");
  Observation result;
  try {
    node.evaluate_array(values, result.out.data(), size);
  } catch (const std::exception& err) {
    result.exception_class = typeid(err).name();
    result.exception_text = err.what();
  }
  result.sticky_flags = std::fetestexcept(FE_ALL_EXCEPT);
  result.rounding = std::fegetround();
  return result;
}

bool bit_equal(double first, double second) {
  return std::bit_cast<std::uint64_t>(first) == std::bit_cast<std::uint64_t>(second);
}

template<class Node>
Observation compare(const Node& baseline, const Node& candidate, const Values& values,
                    int rounding, int preset_flags, size_t size = 4) {
  RestoreFenv restore;
  require(std::fesetround(rounding) == 0 && std::feclearexcept(FE_ALL_EXCEPT) == 0 &&
          std::feraiseexcept(preset_flags) == 0, "prepare the comparison environment");
  fenv_t start;
  require(std::fegetenv(&start) == 0, "capture identical rounding and sticky flags");
  const auto before = evaluate(baseline, values, start, size);
  const auto after = evaluate(candidate, values, start, size);
  require(before.exception_class == after.exception_class && before.exception_text == after.exception_text,
          "exception class, message or precedence differs");
  require(before.rounding == rounding && after.rounding == rounding && before.sticky_flags == after.sticky_flags,
          "rounding mode or sticky flags differ");
  for (size_t i = 0; i < before.out.size(); ++i) {
    if (!bit_equal(before.out[i], after.out[i]))
      throw std::runtime_error("NNJvt output bits differ in component " + std::to_string(i));
  }
  return after;
}

Values inputs(double pt, double eta, double track, double jvf, int vertex = 2, int features = 1) {
  return {pt, eta, track, jvf, std::int64_t(vertex), std::int64_t(features)};
}

std::vector<double> neighborhoods(const rapidjson::Value& edges, float unit) {
  const float infinity = std::numeric_limits<float>::infinity();
  std::vector<double> values;
  for (const auto& edge : edges.GetArray()) {
    const float middle = static_cast<float>(edge.GetDouble()) * unit;
    values.push_back(std::nextafter(middle, -infinity));
    values.push_back(middle);
    values.push_back(std::nextafter(middle, infinity));
  }
  return values;
}

void grid(const correction::Correction& baseline, const correction::Correction& candidate,
          const std::string& config) {
  const double inf = std::numeric_limits<double>::infinity();
  const double qnan = std::bit_cast<double>(UINT64_C(0x7ff8000000012345));
  const double snan = std::bit_cast<double>(UINT64_C(0x7ff0000000012345));
  rapidjson::Document parsed;
  parsed.Parse<rapidjson::kParseFullPrecisionFlag>(config.c_str());
  require(!parsed.HasParseError(), "parse exact fixture configuration");
  const auto& cuts = parsed["cut_map"];
  auto momenta = neighborhoods(cuts["pt_edges_gev"], 1000.f);
  const float cutoff = static_cast<float>(parsed["max_pt"].GetDouble());
  const std::vector<double> other_pt{-inf, -2000., -0., 0., 21'500., 31'500.,
    std::nextafter(cutoff, -std::numeric_limits<float>::infinity()), cutoff,
    std::nextafter(cutoff, std::numeric_limits<float>::infinity()), inf, qnan, snan};
  momenta.insert(momenta.end(), other_pt.begin(), other_pt.end());
  auto etas = neighborhoods(cuts["eta_edges"], 1.f);
  const std::vector<double> other_eta{-inf, -0., 0., 0.37, inf, qnan, snan};
  etas.insert(etas.end(), other_eta.begin(), other_eta.end());
  const double subnormal = std::numeric_limits<float>::denorm_min();
  const std::array tracks{12'500.125, -7.0, 0.0, -0.0, subnormal, inf, qnan};
  const std::array fractions{0.22, -1.0, -0.0, 1.0, -subnormal, inf, qnan, snan};
  std::size_t compared = 0;
  std::size_t evaluated = 0;
  for (int rounding : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
    for (std::size_t i = 0; i < momenta.size(); ++i) {
      for (std::size_t j = 0; j < etas.size(); ++j) {
        for (std::size_t k = 0; k < fractions.size(); ++k) {
          auto args = inputs(momenta[i], etas[j], tracks[(i + j + k) % tracks.size()], fractions[k]);
          const int flags = (k & 1U) != 0U ? FE_DIVBYZERO : 0;
          const auto obs = compare(baseline, candidate, args, rounding, flags);
          require(obs.exception_text.empty(), "real six-argument model unexpectedly threw");
          if (!bit_equal(obs.out[0], -1.0)) ++evaluated;
          ++compared;
        }
      }
    }
  }
  require(evaluated > 1000 && compared - evaluated > 1000, "grid needs actual network and skipped rows");
  std::cout << "public update rows " << compared << ", real network rows " << evaluated
            << ", skipped score rows " << compared - evaluated << '\n';
}

void batch(const correction::Correction& baseline, const correction::Correction& candidate) {
  RestoreFenv restore;
  constexpr std::size_t n = 8;
  const std::array<double, n> pt{21'000., 35'000., 61'000., 45'000., 32'000., 25'000., 56'000., 36'000.};
  const std::array<double, n> eta{0.2, 0.4, -0.9, 3.1, -1.7, 1.4, 0.7, 0.6};
  const std::array<double, n> jvf{0.6, -1., 0., 0.12, 1., -0., 0.3, 0.15};
  const std::array<std::int64_t, n> vertex{2, 2, 2, 2, 0, 1, 2, 2};
  const std::array<std::int64_t, n> features{1, 0, 1, 1, 1, 1, -1, 1};
  using Batch = correction::BatchInput;
  const std::vector<Batch> inputs{
    Batch::reals(pt.data(), n), Batch::reals(eta.data(), n), Batch(12'000.),
    Batch::reals(&jvf.back(), n, -1), Batch::integers(vertex.data(), n), Batch::integers(features.data(), n)
  };
  std::array<double, n * 4> before{}, after{};
  require(std::feclearexcept(FE_ALL_EXCEPT) == 0, "prepare identical batch flags");
  baseline.evaluate_batch_array(inputs, n, before.data(), before.size());
  const auto before_flags = std::fetestexcept(FE_ALL_EXCEPT);
  require(std::feclearexcept(FE_ALL_EXCEPT) == 0, "restore identical batch flags");
  candidate.evaluate_batch_array(inputs, n, after.data(), after.size());
  require(std::fetestexcept(FE_ALL_EXCEPT) == before_flags, "public batched flags");
  for (std::size_t i = 0; i < before.size(); ++i)
    require(bit_equal(before[i], after[i]), "public batched output bits");
}

void gates(const correction::PluginNode& baseline, const correction::PluginNode& candidate) {
  const auto check = [&](const Values& args, size_t size = 4) {
    return compare(baseline, candidate, args, FE_TONEAREST, FE_OVERFLOW, size);
  };
  auto eligible = inputs(30'000.0, 0.25, 15'000.0, 0.32);
  auto normal = check(eligible);
  require(normal.exception_text.empty() && normal.out[0] >= 0.0 &&
          normal.out[1] == 0.5 && normal.out[3] == 1.0, "real prediction must run");

  auto invalid_jvf = eligible;
  invalid_jvf[3] = std::int64_t(2);
  auto malformed = invalid_jvf;
  malformed[5] = std::int64_t(99);
  require(check(malformed).exception_text == "invalid NNJvt input code",
          "feature code error comes before a feature access");
  malformed[4] = std::int64_t(0);
  require(check(malformed).out == Results{-1.0, 0.0, 1.0, -1.0},
          "missing primary vertex prevents feature validation");
  malformed[4] = std::int64_t(1);
  require(check(malformed).out == Results{-1.0, 0.0, 1.0, 1.0},
          "single NoVtx sentinel prevents feature validation");
  malformed[4] = std::int64_t(99);
  require(check(malformed).out == sentinels && check(malformed).exception_text == "invalid NNJvt input code",
          "invalid vertex code precedes all output writes");
  require(check(malformed, 3).exception_text == "NNJvt correction output buffer too small",
          "short output precedes vertex validation");
  malformed[4] = std::int64_t(2);
  for (int feature : {-1, 0}) {
    malformed[5] = std::int64_t(feature);
    require(check(malformed).out == Results{-1.0, 0.0, 1.0, static_cast<double>(feature)},
            "missing features keep the native default decorations");
  }
  malformed[5] = std::int64_t(1);
  malformed[0] = 61'000.0;
  require(check(malformed).exception_text.empty(), "pT veto occurs before JVFCorr decoding");
  malformed[0] = 30'000.0;
  malformed[1] = 4.5;
  require(check(malformed).exception_text.empty(), "eta veto occurs before JVFCorr decoding");
  require(check(eligible).out == normal.out, "eligible call after all exception/skip paths");
}

}  // namespace

int main(int argc, char** argv) {
  require(argc == 5, "usage: nnjvt-inputs-fixture BASELINE_SO CANDIDATE_SO UPDATE_JSON CONFIG_JSON");
  correction::PluginRegistry old_registry, new_registry;
  old_registry.load_library(argv[1]);
  new_registry.load_library(argv[2]);
  auto old_set = correction::CorrectionSet::from_file(argv[3], old_registry);
  auto new_set = correction::CorrectionSet::from_file(argv[3], new_registry);
  const auto old_update = old_set->at("small_nnjvt_update");
  const auto new_update = new_set->at("small_nnjvt_update");
  require(old_update->output_size() == 4 && new_update->output_size() == 4, "update output arity");
  const auto config_json = read(argv[4]);
  const correction::PluginConfig config(config_json);
  const correction::PluginContext old_context(*old_update, {}, 4);
  const correction::PluginContext new_context(*new_update, {}, 4);
  const auto old_raw = old_registry.create("org.atlas.jets.nnjvt_update", 1, config, old_context);
  const auto new_raw = new_registry.create("org.atlas.jets.nnjvt_update", 1, config, new_context);

  void* library = dlopen(argv[2], RTLD_NOW | RTLD_NOLOAD);
  require(library != nullptr, "candidate plugin was physically loaded by the evaluator");
  const auto hold = symbol<const double* (*)()>(library, "nnjvt_fixture_hold");
  const auto release = symbol<bool (*)()>(library, "nnjvt_fixture_release");
  const auto unwind = symbol<bool (*)()>(library, "nnjvt_fixture_unwind");

  auto invalid_jvf = inputs(30'000.0, 0.25, 15'000.0, 0.32);
  invalid_jvf[3] = std::int64_t(2);
  const auto bad_public = compare(*old_update, *new_update, invalid_jvf, FE_TONEAREST, 0);
  require(!bad_public.exception_text.empty() && bad_public.out == sentinels,
          "the public input validator rejects a malformed JVFCorr before invoking the provider");

  // The grid includes the cold first prediction before the test accessor allocates a slot.
  grid(*old_update, *new_update, config_json);
  batch(*old_update, *new_update);
  auto* original_thread = hold();
  require(original_thread && release(), "one 1x4 input slot is available");
  gates(*old_raw, *new_raw);
  require(unwind(), "stack unwinding releases the reusable slot");
  require(hold() == original_thread && release(), "the same scratch is available again");

  require(hold() == original_thread, "outer hold occupies the candidate input slot");
  require(hold() == nullptr, "a second control hold cannot steal the slot");
  gates(*old_raw, *new_raw);
  require(release(), "nested eligible calls preserved the first thread's held values");
  require(hold() == original_thread && release(), "existing slot is reusable after nested calls");

  require(dlclose(library) == 0, "release the fixture's extra handle");
  std::cout << "PASS exact output, gate/exception order, nested and per-thread storage, four rounding modes and sticky flags\n";
}
