#pragma once

#include <rapidjson/document.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace jets::portable {

class Axis {
public:
  Axis(int bins, double low, double high, std::vector<double> edges = {},
       std::optional<std::array<int, 2>> range = std::nullopt);

  int GetNbins() const noexcept { return bins_; }
  double GetXmin() const noexcept { return low_; }
  double GetXmax() const noexcept { return high_; }
  int GetFirst() const noexcept { return first_; }
  int GetLast() const noexcept { return last_; }
  bool has_range() const noexcept { return restricted_; }
  bool has_variable_edges() const noexcept { return !edges_.empty(); }

  int FindFixBin(double x) const noexcept;
  int FindBin(double x) const noexcept { return FindFixBin(x); }
  int legacy_find_bin(double x) const noexcept;
  double GetBinLowEdge(int bin) const noexcept;
  double GetBinUpEdge(int bin) const noexcept;
  double GetBinWidth(int bin) const noexcept;
  double GetBinCenter(int bin) const noexcept;

  Axis projection_axis() const;

private:
  bool variable_interior(int bin) const noexcept;
  int variable_bin(double x) const noexcept;

  int bins_;
  double low_;
  double high_;
  std::vector<double> edges_;
  int first_;
  int last_;
  bool restricted_;
#ifdef JETS_VARIABLE_AXIS_INDEX
  static constexpr int coarse_intervals_ = 64;
  double coarse_scale_ = 0;
  std::vector<std::uint16_t> coarse_bounds_;
#endif
};

class Histogram {
public:
  Histogram(std::string name, std::vector<Axis> axes, std::vector<double> content,
            std::vector<double> sumw2 = {});

  int GetDimension() const noexcept { return static_cast<int>(axes_.size()); }
  const char* ClassName() const noexcept { return name_.c_str(); }
  int GetNcells() const noexcept { return static_cast<int>(content_.size()); }
  const Axis* GetXaxis() const noexcept { return &axes_[0]; }
  const Axis* GetYaxis() const noexcept { return axes_.size() > 1 ? &axes_[1] : nullptr; }
  const Axis* GetZaxis() const noexcept { return axes_.size() > 2 ? &axes_[2] : nullptr; }
  int GetNbinsX() const noexcept { return axes_[0].GetNbins(); }
  int GetNbinsY() const noexcept { return axes_.size() > 1 ? axes_[1].GetNbins() : 1; }
  int GetNbinsZ() const noexcept { return axes_.size() > 2 ? axes_[2].GetNbins() : 1; }
  const Axis& axis(int axis_index) const { return axes_.at(axis_index); }

  int FindBin(double x) const noexcept { return axes_[0].FindFixBin(x); }
  double GetBinCenter(int x) const noexcept { return axes_[0].GetBinCenter(x); }
  int GetBin(int x, int y = 0, int z = 0) const noexcept;
  double GetBinContent(int bin) const noexcept;
  double GetBinContent(int x, int y, int z = 0) const noexcept {
    return GetBinContent(GetBin(x, y, z));
  }
  double GetBinError(int bin) const noexcept;
  double GetBinError(int x, int y, int z = 0) const noexcept {
    return GetBinError(GetBin(x, y, z));
  }
  double Interpolate(double x) const;
  double Interpolate(double x, double y) const;

  Histogram project_x(int y, int z = -1) const;
  Histogram project_y(int x, int z = -1) const;

private:
  Histogram project(int output_axis, int orthogonal_bin, int z) const;

  std::string name_;
  std::vector<Axis> axes_;
  std::vector<double> content_;
  std::vector<double> sumw2_;
};

std::unique_ptr<Histogram> restore_histogram(const rapidjson::Value& data);

double jet_interpolate(const Histogram& hist, double x);
double jet_interpolate(const Histogram& hist, double x, double y);
double jet_interpolate(const Histogram& hist, double x, double y, double z);
double calib_interpolate(const Histogram& hist, double x, double y, double z);

enum class Interpolation { None, Full, OnlyX, OnlyY };

class UncertaintyLookup {
public:
  UncertaintyLookup(std::unique_ptr<Histogram> hist, Interpolation mode);
  const Histogram& histogram() const noexcept { return *hist_; }
  double getValue(double x) const;
  double getValue(double x, double y) const;
  double getValue(double x, double y, double z) const;

private:
  std::unique_ptr<Histogram> hist_;
  Interpolation mode_;
  std::vector<Histogram> projections_;
};

}  // namespace jets::portable
