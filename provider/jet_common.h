#pragma once

#include <array>

#ifdef JETS_PORTABLE
#include "../portable/kinematics.h"
#else
#include <Math/Vector4D.h>
#include <TLorentzVector.h>
#endif

namespace jets {
#ifdef JETS_PORTABLE
using P4 = portable::P4;
using CartesianP4 = portable::CartesianP4;
#else
using P4 = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>;
using CartesianP4 = TLorentzVector;
#endif

inline CartesianP4 as_cartesian(const P4& p) {
  return CartesianP4(p.Px(), p.Py(), p.Pz(), p.E());
}
inline float truth_distance(const CartesianP4& reco, const P4& truth) {
  return static_cast<float>(reco.DeltaR(as_cartesian(truth)));
}

// Jet_v1::setJetP4 writes these float auxiliaries in this order.
inline std::array<double, 4> stored_coordinates(const P4& p) {
  return {static_cast<float>(p.Pt()), static_cast<float>(p.Eta()),
          static_cast<float>(p.Phi()), static_cast<float>(p.M())};
}

// Jet_v1::jetP4 reads the auxiliaries through the coordinate constructor.
inline P4 stored_p4(const P4& p) {
  const auto coordinates = stored_coordinates(p);
  return P4(coordinates[0], coordinates[1], coordinates[2], coordinates[3]);
}

inline P4 relative_pt_mass(double pt, double eta, double phi, double mass,
                           double pt_factor, double mass_factor) {
  return P4(pt_factor * pt, eta, phi, mass_factor * mass);
}
}  // namespace jets
