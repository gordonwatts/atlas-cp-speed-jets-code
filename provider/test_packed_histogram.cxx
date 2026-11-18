// Build against either existing histogram provider. In the portable build
// define JETS_PORTABLE and link portable/histogram.cxx; in the release build
// link provider/histogram.cxx and the installed ROOT libraries.

#include "jet_histogram.h"
#include "packed_histogram.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace encoding = jets::histogram_encoding;

rapidjson::Document parse(const std::string& source) {
  rapidjson::Document doc;
  doc.Parse<rapidjson::kParseFullPrecisionFlag>(source.c_str());
  if (doc.HasParseError()) throw std::runtime_error("Fixture JSON is malformed");
  return doc;
}

std::string encoded(std::string_view base64, std::size_t count) {
  return "{\"encoding\":\"base64-ieee754-binary32-le-v1\",\"count\":" + std::to_string(count) +
         ",\"data\":\"" + std::string(base64) + "\"}";
}

void check_words(std::string_view base64, const std::vector<std::uint32_t>& expected) {
  const auto fields = parse(encoded(base64, expected.size()));
  std::vector<std::uint32_t> actual;
  encoding::decode_binary32(fields, "TH1F", expected.size(), [&](std::size_t i, std::uint32_t word) {
    if (i != actual.size()) throw std::runtime_error("Decoded float word ordering changed");
    actual.push_back(word);
  });
  if (actual != expected) throw std::runtime_error("Raw binary32 cell bits changed");
}

template <class Work>
void must_reject(Work work) {
  try {
    work();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("Packed histogram accepted invalid input");
}

void invalid_data(std::string_view base64, std::size_t claimed_count, std::size_t expected_count,
                  std::string_view root_class = "TH1F") {
  const auto fields = parse(encoded(base64, claimed_count));
  must_reject([&] {
    encoding::decode_binary32(fields, root_class, expected_count, [](std::size_t, std::uint32_t) {});
  });
}

void identical(double decimal, double packed) {
  if (std::bit_cast<std::uint64_t>(decimal) != std::bit_cast<std::uint64_t>(packed))
    throw std::runtime_error("Packed and decimal physics histograms differ");
}

void histogram() {
  const std::string prefix =
      R"({"class":"TH1F","axes":[{"nbins":3,"min":0,"max":6,"edges":[0,1,3,6],"range":[1,3]}],"content":)";
  const std::string suffix = R"(,"sumw2":[4,1,0,9,16]})";
  const auto decimal = parse(prefix + "[-0.0,1,1.401298464324817e-45,-0.75,0.25]" + suffix);
  const auto packed = parse(prefix + encoded("AAAAgAAAgD8BAAAAAABAvwAAgD4=", 5) + suffix);
  const auto a = jets::restore_calibration_histogram(decimal);
  const auto b = jets::restore_calibration_histogram(packed);
  if (std::string_view(a->ClassName()) != b->ClassName() || a->GetNbinsX() != b->GetNbinsX() ||
      a->GetXaxis()->GetFirst() != b->GetXaxis()->GetFirst() || a->GetXaxis()->GetLast() != b->GetXaxis()->GetLast())
    throw std::runtime_error("Packed histogram shape differs");
  for (int bin = 0; bin < a->GetNbinsX() + 2; ++bin) {
    identical(a->GetBinContent(bin), b->GetBinContent(bin));
    identical(a->GetBinError(bin), b->GetBinError(bin));
    identical(a->GetXaxis()->GetBinLowEdge(bin), b->GetXaxis()->GetBinLowEdge(bin));
    identical(a->GetXaxis()->GetBinUpEdge(bin), b->GetXaxis()->GetBinUpEdge(bin));
  }
  if (!std::signbit(b->GetBinContent(0)))
    throw std::runtime_error("Packed TH1F negative-zero flow cell changed");
  const auto infinity = std::numeric_limits<double>::infinity();
  for (double edge : {-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0}) {
    for (double value : {std::nextafter(edge, -infinity), edge, std::nextafter(edge, infinity)})
      identical(a->Interpolate(value), b->Interpolate(value));
  }
}

}  // namespace

int main() {
  // These patterns exercise all three canonical padding widths and preserve
  // signed zero, quiet NaN and signaling NaN before physics validation.
  check_words("AAAAgA==", {0x80000000u});
  check_words("QgDAf5kAgH8=", {0x7fc00042u, 0x7f800099u});
  check_words("AAAAgEIAwH+ZAIB/", {0x80000000u, 0x7fc00042u, 0x7f800099u});
  for (std::uint32_t nonfinite : {0x7f800099u, 0x7fc00042u, 0xff800000u})
    must_reject([&] { static_cast<void>(encoding::finite_binary32(nonfinite)); });
  invalid_data("AAAAgA==", 2, 1);
  invalid_data("AAAAgA==", 1, 1, "TH1D");
  invalid_data("AAAAgA==", 1, 1, "TH2F");
  invalid_data("AAAAgA=", 1, 1);
  invalid_data("AAAAgB==", 1, 1);
  invalid_data("AAAAgA=A", 1, 1);
  invalid_data("A AAAg==", 1, 1);
  invalid_data("QgDAf5kAgH9=", 2, 2);
  const auto unknown = parse(R"({"encoding":"v2","count":1,"data":"AAAAgA=="})");
  must_reject([&] { encoding::decode_binary32(unknown, "TH1F", 1, [](auto, auto) {}); });
  const auto bad_count = parse(R"({"encoding":"base64-ieee754-binary32-le-v1","count":1.0,"data":"AAAAgA=="})");
  must_reject([&] { encoding::decode_binary32(bad_count, "TH1F", 1, [](auto, auto) {}); });
  histogram();
  std::cout << "Exact packed binary32 transport and decimal histogram equality passed\n";
}
