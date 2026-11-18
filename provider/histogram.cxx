#include "histogram.h"
#include "packed_histogram.h"

#include <TH2.h>
#include <TH3.h>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = rapidjson::Value;

const Json& field(const Json& obj, const char* key) {
  if (!obj.IsObject() || !obj.HasMember(key))
    throw std::invalid_argument(std::string("missing histogram field: ") + key);
  return obj[key];
}

double number(const Json& value) {
  if (!value.IsNumber() || !std::isfinite(value.GetDouble()))
    throw std::invalid_argument("expected finite histogram number");
  return value.GetDouble();
}

struct Axis {
  int bins;
  double low;
  double high;
  std::vector<double> edges;
  int first=0, last=0;
  bool restricted=false;
};

Axis read_axis(const Json& value) {
  const auto& count = field(value, "nbins");
  if (!count.IsInt() || count.GetInt() < 1)
    throw std::invalid_argument("invalid histogram axis bin count");
  Axis axis{count.GetInt(), number(field(value, "min")), number(field(value, "max")), {}};
  const auto& edges = field(value, "edges");
  if (axis.low >= axis.high || !edges.IsArray() ||
      (!edges.Empty() && edges.Size() != static_cast<unsigned>(axis.bins) + 1))
    throw std::invalid_argument("invalid histogram axis");
  for (const auto& item : edges.GetArray()) {
    double edge = number(item);
    if (!axis.edges.empty() && edge < axis.edges.back())
      throw std::invalid_argument("histogram edges out of order");
    axis.edges.push_back(edge);
  }
  if (!axis.edges.empty() && (axis.edges.front() != axis.low || axis.edges.back() != axis.high))
    throw std::invalid_argument("histogram axis limits disagree");
  if (value.HasMember("range") && !value["range"].IsNull()) {
    const auto& range=value["range"];
    if (!range.IsArray() || range.Size()!=2 || !range[0].IsInt() || !range[1].IsInt())
      throw std::invalid_argument("invalid ROOT axis range");
    axis.first=range[0].GetInt();
    axis.last=range[1].GetInt();
    axis.restricted=true;
  }
  return axis;
}

template<class H1, class H2, class H3>
std::unique_ptr<TH1> allocate(const std::vector<Axis>& axes) {
  const Axis& x = axes.at(0);
  if (axes.size() == 1) return std::make_unique<H1>("", "", x.bins, x.low, x.high);
  const Axis& y = axes.at(1);
  if (axes.size() == 2)
    return std::make_unique<H2>("", "", x.bins, x.low, x.high, y.bins, y.low, y.high);
  const Axis& z = axes.at(2);
  return std::make_unique<H3>("", "", x.bins, x.low, x.high, y.bins, y.low, y.high, z.bins, z.low, z.high);
}
}  // namespace

std::unique_ptr<TH1> jets::restore_histogram(const rapidjson::Value& data) {
  const auto& kind = field(data, "class");
  const auto& input_axes = field(data, "axes");
  if (!kind.IsString() || !input_axes.IsArray() || input_axes.Empty() || input_axes.Size() > 3)
    throw std::invalid_argument("expected a numeric ROOT histogram of dimension one to three");
  const std::string name = kind.GetString();
  if (name.size() != 4 || name.substr(0, 2) != "TH" || name[2] != '0' + static_cast<int>(input_axes.Size()))
    throw std::invalid_argument("ROOT histogram class/dimension mismatch");
  std::vector<Axis> axes;
  size_t cells = 1;
  for (const auto& item : input_axes.GetArray()) {
    axes.push_back(read_axis(item));
    const size_t width = static_cast<size_t>(axes.back().bins) + 2;
    if (cells > static_cast<size_t>(std::numeric_limits<int>::max()) / width)
      throw std::invalid_argument("histogram exceeds ROOT global-bin limit");
    cells *= width;
  }
  const auto& contents = field(data, "content");
  const auto& sumw2 = field(data, "sumw2");
  const bool packed = contents.IsObject();
  if ((!packed && (!contents.IsArray() || contents.Size() != cells)) ||
      !sumw2.IsArray() || (!sumw2.Empty() && sumw2.Size() != cells))
    throw std::invalid_argument("histogram global bin counts disagree");
  const std::string_view storage_class(kind.GetString(), kind.GetStringLength());
  if (packed && (storage_class != "TH1F" || axes.size() != 1))
    throw std::invalid_argument("packed cells require a one-dimensional TH1F");
  std::unique_ptr<TH1> hist;
  switch (name[3]) {
    case 'C': hist = allocate<TH1C, TH2C, TH3C>(axes); break;
    case 'S': hist = allocate<TH1S, TH2S, TH3S>(axes); break;
    case 'I': hist = allocate<TH1I, TH2I, TH3I>(axes); break;
    case 'F': hist = allocate<TH1F, TH2F, TH3F>(axes); break;
    case 'D': hist = allocate<TH1D, TH2D, TH3D>(axes); break;
    default: throw std::invalid_argument("unsupported ROOT histogram storage class");
  }
  hist->SetDirectory(nullptr);
  std::array<TAxis*, 3> target{hist->GetXaxis(), hist->GetYaxis(), hist->GetZaxis()};
  for (size_t a = 0; a < axes.size(); ++a) {
    if (!axes[a].edges.empty()) target[a]->Set(axes[a].bins, axes[a].edges.data());
    if (axes[a].restricted) {
      target[a]->SetRange(axes[a].first,axes[a].last);
      target[a]->SetBit(TAxis::kAxisRange);
    }
  }
  hist->Sumw2(!sumw2.Empty());
  const auto set_cell = [&](unsigned i, double content) {
    hist->SetBinContent(i, content);
    if (!sumw2.Empty()) hist->GetSumw2()->SetAt(number(sumw2[i]), i);
  };
  if (packed) {
    histogram_encoding::decode_binary32(contents, storage_class, cells, [&](std::size_t i, std::uint32_t word) {
      set_cell(static_cast<unsigned>(i), histogram_encoding::finite_binary32(word));
    });
  } else {
    for (unsigned i = 0; i < cells; ++i) set_cell(i, number(contents[i]));
  }
  return hist;
}
