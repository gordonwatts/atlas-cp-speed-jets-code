#ifndef JETS_CORRECTION_FRAMEWORK_JET_IO_H
#define JETS_CORRECTION_FRAMEWORK_JET_IO_H

#include "JetsCorrectionRuntime/Runtime.h"

#include <AthContainers/Accessor.h>
#include <xAODJet/Jet.h>

#include <string>

namespace jets::framework {

struct MomentumAux {
  const SG::Accessor<float> pt;
  const SG::Accessor<float> eta;
  const SG::Accessor<float> phi;
  const SG::Accessor<float> mass;

  explicit MomentumAux(const std::string& prefix)
      : pt(prefix + "pt"), eta(prefix + "eta"), phi(prefix + "phi"), mass(prefix + "m") {}

  bool available(const xAOD::Jet& jet) const {
    return pt.isAvailable(jet) && eta.isAvailable(jet) && phi.isAvailable(jet) && mass.isAvailable(jet);
  }

  void append(const xAOD::Jet& jet, Values& values) const {
    values.emplace_back(static_cast<double>(pt(jet)));
    values.emplace_back(static_cast<double>(eta(jet)));
    values.emplace_back(static_cast<double>(phi(jet)));
    values.emplace_back(static_cast<double>(mass(jet)));
  }

  void write(xAOD::Jet& jet, const double* values) const {
    pt(jet) = static_cast<float>(values[0]);
    eta(jet) = static_cast<float>(values[1]);
    phi(jet) = static_cast<float>(values[2]);
    mass(jet) = static_cast<float>(values[3]);
  }
};

}  // namespace jets::framework

#endif
