#pragma once

#include <string_view>

#ifdef JETS_PORTABLE
#include "../portable/histogram.h"
#else
#include "histogram.h"

#include <JetCalibTools/RootHelpers.h>
#include <JetUncertainties/JetHelpers.h>
#include <TAxis.h>
#endif

namespace jets {
#ifdef JETS_PORTABLE
using CalibrationHistogram = portable::Histogram;
using CalibrationAxis = portable::Axis;

inline auto restore_calibration_histogram(const rapidjson::Value& data) {
  return portable::restore_histogram(data);
}
using portable::calib_interpolate;
using portable::jet_interpolate;
#else
using CalibrationHistogram = TH1;
using CalibrationAxis = TAxis;

inline auto restore_calibration_histogram(const rapidjson::Value& data) {
  return restore_histogram(data);
}
inline double calib_interpolate(const CalibrationHistogram& histogram, double x, double y, double z) {
  return RootHelpers::Interpolate(&histogram, x, y, z);
}
inline double jet_interpolate(const CalibrationHistogram& histogram, double x, double y) {
  return JetHelpers::Interpolate(&histogram, x, y);
}
inline double jet_interpolate(const CalibrationHistogram& histogram, double x, double y, double z) {
  return JetHelpers::Interpolate(&histogram, x, y, z);
}
#endif

inline bool is_th3f(const CalibrationHistogram& histogram) {
  return std::string_view(histogram.ClassName()) == "TH3F";
}
}  // namespace jets
