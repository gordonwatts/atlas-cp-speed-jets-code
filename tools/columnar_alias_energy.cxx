#include "../provider/jet_common.h"

#include <cstddef>

// AnalysisBase 25.2.117 AsgEnergyDecoratorAlg stores particle->e() in a
// float. xAOD::Jet_v1::e() returns jetP4().E() from float aux coordinates.
extern "C" void columnar_alias_energy(std::size_t size, const float* pt, const float* eta,
                                      const float* phi, const float* mass, float* energy) {
  for (std::size_t i = 0; i < size; ++i) {
    energy[i] = static_cast<float>(jets::P4(pt[i], eta[i], phi[i], mass[i]).E());
  }
}
