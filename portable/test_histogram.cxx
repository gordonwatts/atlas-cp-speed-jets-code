#include "histogram.h"

#include <JetCalibTools/RootHelpers.h>
#include <JetUncertainties/JetHelpers.h>
#include <JetUncertainties/UncertaintyHistogram.h>
#include <TError.h>
#include <TH2.h>
#include <TH3.h>
#include <TMemFile.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace portable = jets::portable;

// The compiled release helper exports this testable function without a header declaration.
namespace RootHelpers { int FindBin(const TAxis*, double); }

namespace {

std::size_t checks = 0;

void equal(double expected, double got, std::string_view field) {
  ++checks;
  if (std::bit_cast<std::uint64_t>(expected) == std::bit_cast<std::uint64_t>(got) ||
      (std::isnan(expected) && std::isnan(got)))
    return;
  std::cerr << std::setprecision(17) << field << ": expected " << expected
            << ", got " << got << '\n';
  throw std::runtime_error("release-oracle difference");
}

auto axis_from_root(const TAxis& source) {
  std::vector<double> edges;
  for (int i = 0; i < source.GetXbins()->GetSize(); ++i) edges.push_back(source.GetXbins()->At(i));
  std::optional<std::array<int, 2>> range;
  if (source.TestBit(TAxis::kAxisRange))
    range = std::array{source.GetFirst(), source.GetLast()};
  return portable::Axis(source.GetNbins(), source.GetXmin(), source.GetXmax(), edges, range);
}

portable::Histogram from_root(const TH1& source) {
  std::vector<portable::Axis> axes{axis_from_root(*source.GetXaxis())};
  if (source.GetDimension() > 1) axes.push_back(axis_from_root(*source.GetYaxis()));
  if (source.GetDimension() > 2) axes.push_back(axis_from_root(*source.GetZaxis()));
  std::vector<double> cells, variances;
  for (int i = 0; i < source.GetNcells(); ++i) {
    cells.push_back(source.GetBinContent(i));
    if (source.GetSumw2N()) variances.push_back(source.GetSumw2()->At(i));
  }
  return {source.ClassName(), std::move(axes), std::move(cells), std::move(variances)};
}

std::vector<double> points(const TAxis& axis) {
  std::vector<double> values;
  const double inf = std::numeric_limits<double>::infinity();
  auto include = [&](double x) {
    values.insert(values.end(), {std::nextafter(x, -inf), x, std::nextafter(x, inf)});
  };
  for (int bin = 1; bin <= axis.GetNbins(); ++bin) {
    include(axis.GetBinLowEdge(bin));
    include(axis.GetBinCenter(bin));
    if (bin == axis.GetNbins()) include(axis.GetBinUpEdge(bin));
  }
  return values;
}

void check_geometry(const TAxis& root, const portable::Axis& p) {
  equal(root.GetFirst(), p.GetFirst(), "axis first");
  equal(root.GetLast(), p.GetLast(), "axis last");
  for (int bin = -1; bin <= root.GetNbins() + 2; ++bin) {
    equal(root.GetBinLowEdge(bin), p.GetBinLowEdge(bin), "low edge");
    equal(root.GetBinUpEdge(bin), p.GetBinUpEdge(bin), "up edge");
    equal(root.GetBinWidth(bin), p.GetBinWidth(bin), "width");
    equal(root.GetBinCenter(bin), p.GetBinCenter(bin), "center");
  }
  auto samples = points(root);
  samples.push_back(std::numeric_limits<double>::quiet_NaN());
  for (double x : samples) {
    equal(root.FindFixBin(x), p.FindFixBin(x), "fixed bin");
    equal(RootHelpers::FindBin(&root, x), p.legacy_find_bin(x), "legacy bin");
  }
}

void check_cells(const TH1& root, const portable::Histogram& p) {
  equal(root.GetNcells(), p.GetNcells(), "global cells");
  for (int bin = -1; bin <= root.GetNcells(); ++bin) {
    equal(root.GetBinContent(bin), p.GetBinContent(bin), "cell content");
    equal(root.GetBinError(bin), p.GetBinError(bin), "cell error");
  }
}

void fill(TH1& source, bool store_errors) {
  if (store_errors) source.Sumw2();
  for (int bin = 0; bin < source.GetNcells(); ++bin) {
    source.SetBinContent(bin, std::sin(0.71 * bin) * (bin + 1) * 0.131);
    if (store_errors) source.GetSumw2()->SetAt(0.17 + (bin % 11) * 0.37, bin);
  }
}

void check_lookup(TH1& root, const portable::Histogram& p) {
  TMemFile file("portable_proof.root", "RECREATE");
  root.Write("sample");
  using Native = jet::Interpolate::TypeEnum;
  const std::array options{
    std::pair<Native, portable::Interpolation>{jet::Interpolate::None, portable::Interpolation::None},
    std::pair<Native, portable::Interpolation>{jet::Interpolate::Full, portable::Interpolation::Full},
    std::pair<Native, portable::Interpolation>{jet::Interpolate::OnlyX, portable::Interpolation::OnlyX},
    std::pair<Native, portable::Interpolation>{jet::Interpolate::OnlyY, portable::Interpolation::OnlyY}
  };
  const auto xs = points(*root.GetXaxis());
  const auto ys = root.GetDimension() > 1 ? points(*root.GetYaxis()) : std::vector<double>{0};
  const auto zs = root.GetDimension() > 2 ? points(*root.GetZaxis()) : std::vector<double>{0};

  for (const auto& [stock, port] : options) {
    if (root.GetDimension() == 1 && port == portable::Interpolation::OnlyY) continue;
    jet::UncertaintyHistogram oracle("sample", stock);
    if (!oracle.initialize(&file).isSuccess()) throw std::runtime_error("stock lookup init failed");
    portable::UncertaintyLookup actual(std::make_unique<portable::Histogram>(p), port);
    for (double x : xs)
      for (double y : ys)
        for (double z : zs) {
          if (root.GetDimension() == 1)
            equal(oracle.getValue(x), actual.getValue(x), "1D lookup");
          else if (root.GetDimension() == 2)
            equal(oracle.getValue(x, y), actual.getValue(x, y), "2D lookup");
          else
            equal(oracle.getValue(x, y, z), actual.getValue(x, y, z), "3D lookup");
        }
  }
}

void check_histogram(TH1& root) {
  const auto p = from_root(root);
  check_cells(root, p);
  check_geometry(*root.GetXaxis(), *p.GetXaxis());
  const auto xs = points(*root.GetXaxis());
  if (root.GetDimension() == 1) {
    for (double x : xs) equal(root.Interpolate(x), p.Interpolate(x), "TH1 interpolation");
  } else {
    check_geometry(*root.GetYaxis(), *p.GetYaxis());
    const auto ys = points(*root.GetYaxis());
    if (root.GetDimension() == 2) {
      for (double x : xs)
        for (double y : ys) {
          equal(root.Interpolate(x, y), p.Interpolate(x, y), "TH2 interpolation");
          equal(JetHelpers::Interpolate(&root, x, y), portable::jet_interpolate(p, x, y),
                "JetHelpers 2D interpolation");
        }
      const auto& h = dynamic_cast<const TH2&>(root);
      for (int y = 0; y <= h.GetNbinsY() + 1; ++y) {
        std::unique_ptr<TH1D> projection(h.ProjectionX("px_fixture", y, y));
        check_cells(*projection, p.project_x(y));
      }
      for (int x = 0; x <= h.GetNbinsX() + 1; ++x) {
        std::unique_ptr<TH1D> projection(h.ProjectionY("py_fixture", x, x));
        check_cells(*projection, p.project_y(x));
      }
    } else {
      check_geometry(*root.GetZaxis(), *p.GetZaxis());
      for (double x : xs)
        for (double y : ys)
          for (double z : points(*root.GetZaxis())) {
            equal(JetHelpers::Interpolate(&root, x, y, z),
                  portable::jet_interpolate(p, x, y, z), "JetHelpers 3D interpolation");
            equal(RootHelpers::Interpolate(&root, x, y, z),
                  portable::calib_interpolate(p, x, y, z), "calibration 3D interpolation");
          }
      const auto& h = dynamic_cast<const TH3&>(root);
      for (int z = 0; z <= h.GetNbinsZ() + 1; ++z) {
        for (int y = 0; y <= h.GetNbinsY() + 1; ++y) {
          std::unique_ptr<TH1D> projection(h.ProjectionX("p3x_fixture", y, y, z, z));
          check_cells(*projection, p.project_x(y, z));
        }
        for (int x = 0; x <= h.GetNbinsX() + 1; ++x) {
          std::unique_ptr<TH1D> projection(h.ProjectionY("p3y_fixture", x, x, z, z));
          check_cells(*projection, p.project_y(x, z));
        }
      }
    }
  }
  check_lookup(root, p);
}

} // namespace

int main() {
  TH1::AddDirectory(false);
  gErrorIgnoreLevel = kFatal;
  TH1D one("root_one", "", 6, -0.39, 0.84);
  TH1F singleton("root_single", "", 1, 0.2, 0.7);
  const double x_edges[] = {0.2, 0.39, 1.2, 6.8};
  const double y_edges[] = {-1.7, 0.2, 0.53, 1.4};
  const double z_edges[] = {-2.3, -0.24, 0.47, 1.2};
  TH2F two("root_two", "", 3, x_edges, 3, y_edges);
  two.GetXaxis()->SetRange(2, 3);
  TH3F full("root_three", "", 3, x_edges, 3, y_edges, 3, z_edges);
  TH3F flat("root_flat", "", 4, -0.33, 1.2, 1, 0.1, 1.1, 3, -0.4, 2.3);
  flat.GetXaxis()->SetRange(2, 3);

  fill(one, true);
  fill(singleton, false);
  fill(two, true);
  fill(full, true);
  fill(flat, false);
  for (TH1* hist : {static_cast<TH1*>(&one), static_cast<TH1*>(&singleton),
                    static_cast<TH1*>(&two), static_cast<TH1*>(&full), static_cast<TH1*>(&flat)})
    check_histogram(*hist);

  rapidjson::Document json;
  json.Parse(R"({
      "class":"TH1F","axes":[{"nbins":2,"min":0,"max":4,"edges":[0,1,4],"range":[2,2]}],
      "content":[-1,0.1,4,5],"sumw2":[0.25,0.81,0.64,4]
  })");
  auto hist = portable::restore_histogram(json);
  equal(static_cast<double>(static_cast<float>(0.1)), hist->GetBinContent(1), "JSON payload float cell");
  equal(std::sqrt(0.64), hist->GetBinError(2), "JSON payload error");
  equal(2, hist->GetXaxis()->GetFirst(), "JSON payload range");
  std::cout << "exact release comparisons: " << checks << '\n';
}
