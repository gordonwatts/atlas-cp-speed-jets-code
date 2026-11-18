#include "histogram.h"

#include "../provider/packed_histogram.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace jets::portable {
namespace {

// ROOT 6.40.02 TAxis and TH1/TH2/TH3 numerical behavior.
// https://github.com/root-project/root/tree/v6-40-02/hist/hist/src
// The ATLAS variants follow AnalysisBase 25.2.117 JetUncertainties/JetHelpers
// and JetCalibTools/RootHelpers. In particular, their 3D widths differ.

using Json = rapidjson::Value;

const Json& field(const Json& data, const char* key) {
  if (!data.IsObject() || !data.HasMember(key))
    throw std::invalid_argument(std::string("missing histogram field: ") + key);
  return data[key];
}

double number(const Json& value) {
  if (!value.IsNumber() || !std::isfinite(value.GetDouble()))
    throw std::invalid_argument("expected finite histogram number");
  return value.GetDouble();
}

std::vector<double> number_array(const Json& value) {
  if (!value.IsArray()) throw std::invalid_argument("expected histogram number array");
  std::vector<double> values;
  values.reserve(value.Size());
  for (const auto& cell : value.GetArray()) values.push_back(number(cell));
  return values;
}

Axis read_axis(const Json& data) {
  const auto& size = field(data, "nbins");
  if (!size.IsInt()) throw std::invalid_argument("invalid histogram axis bin count");
  std::optional<std::array<int, 2>> range;
  if (data.HasMember("range") && !data["range"].IsNull()) {
    const auto& r = data["range"];
    if (!r.IsArray() || r.Size() != 2 || !r[0].IsInt() || !r[1].IsInt())
      throw std::invalid_argument("invalid histogram axis range");
    range = std::array{r[0].GetInt(), r[1].GetInt()};
  }
  return Axis(size.GetInt(), number(field(data, "min")), number(field(data, "max")),
              number_array(field(data, "edges")), range);
}

template<class T>
double integral_cell(double value) {
  if (!std::isfinite(value) || value < std::numeric_limits<T>::min() || value > std::numeric_limits<T>::max())
    throw std::invalid_argument("histogram cell outside declared integer storage");
  return static_cast<T>(value);
}

double stored_cell(char storage, double value) {
  switch (storage) {
    case 'C': return integral_cell<signed char>(value);
    case 'S': return integral_cell<short>(value);
    case 'I': return integral_cell<int>(value);
    case 'F': return static_cast<float>(value);
    case 'D': return value;
    default: throw std::invalid_argument("unsupported histogram storage class");
  }
}

struct Centers {
  int low_bin;
  int high_bin;
  double low;
  double high;
};

using FindBin = int (Axis::*)(double) const noexcept;

Centers neighboring_centers(const Axis& axis, double x, int bin, FindBin find_bin) {
  // For the ROOT bilinear implementation, use the comparison to the upper
  // edge and half the width before resolving the neighbors by their centers.
  const int low = bin - (axis.GetBinUpEdge(bin) - x > axis.GetBinWidth(bin) / 2);
  const double left = axis.GetBinCenter(low);
  const double right = axis.GetBinCenter(low + 1);
  return {std::max((axis.*find_bin)(left), 1),
          std::min((axis.*find_bin)(right), axis.GetNbins()), left, right};
}

double interpolate_2d(const Histogram& hist, double x, double y,
                      int x_axis, int y_axis, std::optional<int> fixed_bin,
                      FindBin find_bin) {
  const Axis& x_geometry = hist.axis(x_axis);
  const Axis& y_geometry = hist.axis(y_axis);
  const int x_bin = (x_geometry.*find_bin)(x);
  const int y_bin = (y_geometry.*find_bin)(y);
  if (x_bin < 1 || x_bin > x_geometry.GetNbins() ||
      y_bin < 1 || y_bin > y_geometry.GetNbins())
    return 0;

  const auto a = neighboring_centers(x_geometry, x, x_bin, find_bin);
  const auto b = neighboring_centers(y_geometry, y, y_bin, find_bin);
  std::array<int, 3> bins{};
  if (fixed_bin) bins[3 - x_axis - y_axis] = *fixed_bin;
  auto content = [&](int bin_a, int bin_b) {
    bins[x_axis] = bin_a;
    bins[y_axis] = bin_b;
    return hist.GetBinContent(bins[0], bins[1], bins[2]);
  };
  const double q11 = content(a.low_bin, b.low_bin);
  const double q21 = content(a.high_bin, b.low_bin);
  const double q12 = content(a.low_bin, b.high_bin);
  const double q22 = content(a.high_bin, b.high_bin);
  const double area = (a.high - a.low) * (b.high - b.low);
  return q11 / area * (a.high - x) * (b.high - y) +
         q21 / area * (x - a.low) * (b.high - y) +
         q12 / area * (a.high - x) * (y - b.low) +
         q22 / area * (x - a.low) * (y - b.low);
}

double interpolate_3d(const Histogram& hist, std::array<double, 3> xyz, bool legacy_width) {
  if (hist.GetDimension() != 3) return 0;
  const FindBin find_bin = legacy_width ? &Axis::legacy_find_bin : &Axis::FindFixBin;
  std::array<int, 3> below{}, above{};
  std::array<double, 3> widths{};
  for (int dimension = 0; dimension < 3; ++dimension) {
    const Axis& axis = hist.axis(dimension);
    const int bin = (axis.*find_bin)(xyz[dimension]);
    if (bin < 1 || bin > axis.GetNbins()) return 0;
    if (legacy_width)
      widths[dimension] = axis.GetBinCenter(bin + 1) - axis.GetBinCenter(bin);
    const int lower = bin - (xyz[dimension] < axis.GetBinCenter(bin));
    below[dimension] = std::max(lower, 1);
    above[dimension] = std::min(lower + 1, axis.GetNbins());
    if (!legacy_width)
      widths[dimension] = axis.GetBinCenter(above[dimension]) - axis.GetBinCenter(below[dimension]);
  }

  // The release tries the Z, Y and X boundary cases in that order. The
  // 2D rule can still interpolate when either remaining axis is a singleton.
  for (int fixed = 2; fixed >= 0; --fixed) {
    if (below[fixed] == above[fixed]) {
      const int first = fixed == 0 ? 1 : 0;
      const int second = fixed == 2 ? 1 : 2;
      return interpolate_2d(hist, xyz[first], xyz[second], first, second,
                            below[fixed], find_bin);
    }
  }

  std::array<double, 3> fraction{};
  for (int a = 0; a < 3; ++a)
    fraction[a] = (xyz[a] - hist.axis(a).GetBinCenter(below[a])) / widths[a];
  const double tx = fraction[0], ty = fraction[1], tz = fraction[2];
  auto across_z = [&](int x, int y) {
    return hist.GetBinContent(x, y, below[2]) * (1 - tz) +
           hist.GetBinContent(x, y, above[2]) * tz;
  };
  const double a = across_z(below[0], below[1]) * (1 - ty) +
                   across_z(below[0], above[1]) * ty;
  const double b = across_z(above[0], below[1]) * (1 - ty) +
                   across_z(above[0], above[1]) * ty;
  return a * (1 - tx) + b * tx;
}

double uncertainty_coordinate(const Axis& axis, double value) {
#ifdef JETS_FAST_VARIABLE_FLOW
  if (axis.has_variable_edges()) {
    // Sorted variable edges, including duplicates, give an interior bin
    // for every value in [xmin, xmax). Only flow classification is needed.
    if (value < axis.GetXmin()) return axis.GetBinLowEdge(1);
    if (value < axis.GetXmax()) return value;
  } else
#endif
  {
    const int bin = axis.FindFixBin(value);
    if (bin == 0) return axis.GetBinLowEdge(1);
    if (bin <= axis.GetNbins()) return value;
  }
  return (1. - 1.e-3) * axis.GetBinWidth(axis.GetNbins()) + axis.GetBinLowEdge(axis.GetNbins());
}

}  // namespace

Axis::Axis(int bins, double low, double high, std::vector<double> edges,
           std::optional<std::array<int, 2>> range)
    : bins_(bins), low_(low), high_(high), edges_(std::move(edges)),
      first_(1), last_(bins), restricted_(range.has_value()) {
  if (bins < 1 || bins > std::numeric_limits<int>::max() - 2 ||
      !std::isfinite(low) || !std::isfinite(high) || low >= high ||
      !std::isfinite(high - low) || (high - low) / bins == 0.0)
    throw std::invalid_argument("invalid histogram axis");
  if (!edges_.empty() &&
      (edges_.size() != static_cast<std::size_t>(bins) + 1 ||
       edges_.front() != low || edges_.back() != high ||
       !std::is_sorted(edges_.begin(), edges_.end()) ||
       !std::all_of(edges_.begin(), edges_.end(), [](double edge) { return std::isfinite(edge); })))
    throw std::invalid_argument("invalid histogram variable edges");
  if (range) {
    const auto [first, last] = *range;
    if (first <= last && last >= 0 && first <= bins + 1 && (first || last)) {
      first_ = std::max(first, 0);
      last_ = std::min(last, bins + 1);
    }
  }
}

bool Axis::variable_interior(int bin) const noexcept {
  return !edges_.empty() && bin >= 1 && bin <= bins_;
}

int Axis::variable_bin(double x) const noexcept {
  const auto edge = std::lower_bound(edges_.begin(), edges_.end(), x);
  return static_cast<int>(edge - edges_.begin()) + (edge != edges_.end() && *edge == x);
}

int Axis::FindFixBin(double x) const noexcept {
  if (x < low_) return 0;
  if (!(x < high_)) return bins_ + 1;
  if (!edges_.empty()) return variable_bin(x);
  const double width = (high_ - low_) / bins_;
  const int estimate = static_cast<int>((x - low_) / width);
  return 1 + estimate - (x < low_ + width * estimate) + (low_ + width * (estimate + 1) <= x);
}

int Axis::legacy_find_bin(double x) const noexcept {
  if (x < low_) return 0;
  if (!(x < high_)) return bins_ + 1;
  if (!edges_.empty()) return variable_bin(x);
  return 1 + static_cast<int>(bins_ * (x - low_) / (high_ - low_));
}

double Axis::GetBinLowEdge(int bin) const noexcept {
  if (variable_interior(bin)) return edges_[bin - 1];
  return low_ + (bin - 1) * ((high_ - low_) / bins_);
}

double Axis::GetBinUpEdge(int bin) const noexcept {
  if (variable_interior(bin)) return edges_[bin];
  return low_ + bin * ((high_ - low_) / bins_);
}

double Axis::GetBinWidth(int bin) const noexcept {
  if (variable_interior(bin)) return edges_[bin] - edges_[bin - 1];
  return (high_ - low_) / bins_;
}

double Axis::GetBinCenter(int bin) const noexcept {
  if (variable_interior(bin)) return edges_[bin - 1] + 0.5 * GetBinWidth(bin);
  return low_ + (bin - 0.5) * ((high_ - low_) / bins_);
}

Axis Axis::projection_axis() const {
  const int first = std::max(first_, 1);
  const int last = std::min(last_, bins_);
  std::vector<double> edges;
  if (!edges_.empty())
    edges.assign(edges_.begin() + first - 1, edges_.begin() + last + 1);
  return Axis(last - first + 1, GetBinLowEdge(first), GetBinUpEdge(last), std::move(edges));
}

Histogram::Histogram(std::string name, std::vector<Axis> axes,
                     std::vector<double> content, std::vector<double> sumw2)
    : name_(std::move(name)), axes_(std::move(axes)), content_(std::move(content)),
      sumw2_(std::move(sumw2)) {
  if (axes_.empty() || axes_.size() > 3 || name_.size() != 4 || name_.substr(0, 2) != "TH" ||
      name_[2] != '0' + GetDimension() || std::string("CSIFD").find(name_[3]) == std::string::npos)
    throw std::invalid_argument("histogram class and dimension disagree");
  std::size_t cells = 1;
  for (const Axis& axis : axes_) {
    const std::size_t width = axis.GetNbins() + 2;
    if (cells > static_cast<std::size_t>(std::numeric_limits<int>::max()) / width)
      throw std::invalid_argument("histogram exceeds ROOT global-bin limit");
    cells *= width;
  }
  if (content_.size() != cells || (!sumw2_.empty() && sumw2_.size() != cells))
    throw std::invalid_argument("histogram global bin counts disagree");
  if (name_[3] != 'D')
    for (double& cell : content_) cell = stored_cell(name_[3], cell);
}

std::unique_ptr<Histogram> restore_histogram(const Json& data) {
  const auto& name = field(data, "class");
  const auto& json_axes = field(data, "axes");
  if (!name.IsString() || !json_axes.IsArray() || json_axes.Empty() || json_axes.Size() > 3)
    throw std::invalid_argument("expected a numeric histogram of dimension one to three");
  std::vector<Axis> axes;
  for (const auto& value : json_axes.GetArray()) axes.push_back(read_axis(value));
  const auto& content = field(data, "content");
  std::vector<double> cells;
  if (content.IsObject()) {
    const auto kind = std::string_view(name.GetString(), name.GetStringLength());
    // The packed format is defined only for the one-dimensional TH1F.
    if (axes.size() != 1) throw std::invalid_argument("packed histogram dimension mismatch");
    const auto count = static_cast<std::size_t>(axes.front().GetNbins()) + 2;
    histogram_encoding::decode_binary32(content, kind, count, [&](std::size_t i, std::uint32_t word) {
      if (i == 0) cells.reserve(count);
      cells.push_back(histogram_encoding::finite_binary32(word));
    });
  } else {
    cells = number_array(content);
  }
  return std::make_unique<Histogram>(name.GetString(), std::move(axes), std::move(cells),
                                     number_array(field(data, "sumw2")));
}

int Histogram::GetBin(int x, int y, int z) const noexcept {
  const int first = std::clamp(x, 0, GetNbinsX() + 1);
  if (axes_.size() == 1) return first;
  const int second = std::clamp(y, 0, GetNbinsY() + 1);
  if (axes_.size() == 2) return first + (GetNbinsX() + 2) * second;
  const int third = std::clamp(z, 0, GetNbinsZ() + 1);
  return first + (GetNbinsX() + 2) * (second + (GetNbinsY() + 2) * third);
}

double Histogram::GetBinContent(int bin) const noexcept {
  return content_[std::clamp(bin, 0, GetNcells() - 1)];
}

double Histogram::GetBinError(int bin) const noexcept {
  bin = std::clamp(bin, 0, GetNcells() - 1);
  return std::sqrt(sumw2_.empty() ? std::abs(content_[bin]) : sumw2_[bin]);
}

double Histogram::Interpolate(double x) const {
  if (GetDimension() != 1) return 0;
  if (std::isnan(x)) return x;  // ROOT TH1 has no defined out-of-array NaN retrieval.
  if (x <= GetBinCenter(1)) return GetBinContent(1);
  if (x >= GetBinCenter(GetNbinsX())) return GetBinContent(GetNbinsX());
  const int bin = axes_[0].FindFixBin(x);
  const int left = bin - (x <= GetBinCenter(bin));
  const double start = GetBinContent(left);
  const double center = GetBinCenter(left);
  return start + (x - center) * ((GetBinContent(left + 1) - start) / (GetBinCenter(left + 1) - center));
}

double Histogram::Interpolate(double x, double y) const {
  if (GetDimension() != 2) return 0;
  return interpolate_2d(*this, x, y, 0, 1, std::nullopt, &Axis::FindFixBin);
}

Histogram Histogram::project(int output_axis, int other_bin, int z) const {
  if (GetDimension() < 2 || (GetDimension() == 2 && z != -1) ||
      (GetDimension() == 3 && (z < 0 || z > GetNbinsZ() + 1)) ||
      other_bin < 0 || other_bin > axis(1 - output_axis).GetNbins() + 1)
    throw std::invalid_argument("invalid projection slice");
  const Axis& source_axis = axis(output_axis);
  Axis projected = source_axis.projection_axis();
  std::vector<double> values(projected.GetNbins() + 2);
  std::vector<double> variances(sumw2_.empty() ? 0 : values.size());
  // TH3::ProjectionX/Y goes through TAxis::SetRange. A request of (0,0)
  // resets the integrated axis, so it sums all bins including both flows.
  const bool three = GetDimension() == 3;
  const int integrated_max = three && other_bin == 0 ? axis(1 - output_axis).GetNbins() + 1 : other_bin;
  const int z_begin = three ? z : 0;
  const int z_end = three && z == 0 ? GetNbinsZ() + 1 : z_begin;
  for (int out_bin = 0; out_bin <= source_axis.GetNbins() + 1; ++out_bin) {
    if (source_axis.has_range() &&
        (out_bin < std::max(source_axis.GetFirst(), 1) ||
         out_bin > std::min(source_axis.GetLast(), source_axis.GetNbins())))
      continue;

    double contents = 0;
    double variance = 0;
    for (int integrated = other_bin; integrated <= integrated_max; ++integrated)
      for (int zbin = z_begin; zbin <= z_end; ++zbin) {
        const int cell = output_axis == 0 ? GetBin(out_bin, integrated, zbin)
                                          : GetBin(integrated, out_bin, zbin);
        contents += GetBinContent(cell);
        if (!variances.empty()) {
          const double error = GetBinError(cell);
          variance += error * error;
        }
      }
    const int target = projected.FindFixBin(source_axis.GetBinCenter(out_bin));
    values[target] = contents;
    if (!variances.empty()) {
      const double error = std::sqrt(variance);
      variances[target] = error * error;
    }
  }
  return Histogram("TH1D", {std::move(projected)}, std::move(values), std::move(variances));
}

Histogram Histogram::project_x(int y, int z) const { return project(0, y, z); }
Histogram Histogram::project_y(int x, int z) const { return project(1, x, z); }

double jet_interpolate(const Histogram& hist, double x) { return hist.Interpolate(x); }
double jet_interpolate(const Histogram& hist, double x, double y) { return hist.Interpolate(x, y); }
double jet_interpolate(const Histogram& hist, double x, double y, double z) {
  return interpolate_3d(hist, {x, y, z}, false);
}
double calib_interpolate(const Histogram& hist, double x, double y, double z) {
  return interpolate_3d(hist, {x, y, z}, true);
}

UncertaintyLookup::UncertaintyLookup(std::unique_ptr<Histogram> hist, Interpolation mode)
    : hist_(std::move(hist)), mode_(mode) {
  if (!hist_) throw std::invalid_argument("missing uncertainty histogram");
  if (hist_->GetDimension() == 1) {
    if (mode_ == Interpolation::OnlyY)
      throw std::invalid_argument("Y interpolation requires a Y axis");
    return;
  }
  if (mode_ != Interpolation::OnlyX && mode_ != Interpolation::OnlyY) return;
  const bool only_x = mode_ == Interpolation::OnlyX;
  const int count = only_x ? hist_->GetNbinsY() : hist_->GetNbinsX();
  const bool three = hist_->GetDimension() == 3;
  const int z_count = three ? hist_->GetNbinsZ() + 1 : 1;
  projections_.reserve(static_cast<std::size_t>(count + 1) * z_count);
  for (int index = 0; index <= count; ++index)
    for (int z = 0; z < z_count; ++z) {
      projections_.push_back(only_x ? hist_->project_x(index, three ? z : -1)
                                    : hist_->project_y(index, three ? z : -1));
    }
}

double UncertaintyLookup::getValue(double x) const {
  if (hist_->GetDimension() != 1) return 0;
  const double xf = uncertainty_coordinate(hist_->axis(0), x);
  return mode_ == Interpolation::None ? hist_->GetBinContent(hist_->FindBin(xf))
                                      : jet_interpolate(*hist_, xf);
}

double UncertaintyLookup::getValue(double x, double y) const {
  if (hist_->GetDimension() != 2) return 0;
  const Axis& xa = hist_->axis(0);
  const Axis& ya = hist_->axis(1);
  const double xf = uncertainty_coordinate(xa, x);
  const double yf = uncertainty_coordinate(ya, y);
  switch (mode_) {
    case Interpolation::Full: return jet_interpolate(*hist_, xf, yf);
    case Interpolation::OnlyX: return jet_interpolate(projections_.at(ya.FindFixBin(yf)), xf);
    case Interpolation::OnlyY: return jet_interpolate(projections_.at(xa.FindFixBin(xf)), yf);
    case Interpolation::None: return hist_->GetBinContent(xa.FindFixBin(xf), ya.FindFixBin(yf));
  }
  throw std::invalid_argument("unsupported uncertainty interpolation");
}

double UncertaintyLookup::getValue(double x, double y, double z) const {
  if (hist_->GetDimension() != 3) return 0;
  const Axis& xa = hist_->axis(0);
  const Axis& ya = hist_->axis(1);
  const Axis& za = hist_->axis(2);
  const double xf = uncertainty_coordinate(xa, x);
  const double yf = uncertainty_coordinate(ya, y);
  const double zf = uncertainty_coordinate(za, z);
  const auto stride = static_cast<std::size_t>(za.GetNbins()) + 1;
  switch (mode_) {
    case Interpolation::Full: return jet_interpolate(*hist_, xf, yf, zf);
    case Interpolation::OnlyX:
    case Interpolation::OnlyY: {
      const bool only_x = mode_ == Interpolation::OnlyX;
      const auto z_bin = static_cast<std::size_t>(za.FindFixBin(zf));
      if (z_bin >= stride) throw std::out_of_range("uncertainty projection z bin");
      const auto orthogonal = only_x ? ya.FindFixBin(yf) : xa.FindFixBin(xf);
      return jet_interpolate(projections_.at(orthogonal * stride + z_bin), only_x ? xf : yf);
    }
    case Interpolation::None:
      return hist_->GetBinContent(xa.FindFixBin(xf), ya.FindFixBin(yf), za.FindFixBin(zf));
  }
  throw std::invalid_argument("unsupported uncertainty interpolation");
}

}  // namespace jets::portable
