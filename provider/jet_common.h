#pragma once

#include <Math/Vector4D.h>
#include <TLorentzVector.h>

namespace jets {
using P4 = ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>;

inline TLorentzVector as_cartesian(const P4& p) {
  return TLorentzVector(p.Px(), p.Py(), p.Pz(), p.E());
}
inline float truth_distance(const TLorentzVector& reco, const P4& truth) {
  return static_cast<float>(reco.DeltaR(as_cartesian(truth)));
}

// Each Jet_v1 assignment writes float auxiliaries. Read them back through the
// same ROOT coordinate constructor as xAOD::Jet_v1::jetP4().
inline P4 stored_p4(const P4& p) {
  return P4(static_cast<float>(p.Pt()), static_cast<float>(p.Eta()),
            static_cast<float>(p.Phi()), static_cast<float>(p.M()));
}
}  // namespace jets
