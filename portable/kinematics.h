// SPDX-License-Identifier: LGPL-2.1-or-later
// Derived from ROOT 6.40.02 GenVector, TLorentzVector and TVector3.
// ROOT authors' notices and source locations are recorded in ROOT-NOTICE.md.

#pragma once

#include <cmath>
#include <limits>

namespace jets::portable {

inline constexpr double pi = 3.14159265358979323846;
inline constexpr double eta_max = 22756.0;

inline double cylindrical_eta(double transverse, double z) {
  if (transverse > 0.0) {
    // ROOT GenVector uses a first-order large-slope expansion. Replacing
    // this expression with asinh changes float writeback at real jet values.
    static const double expansion_threshold = std::pow(std::numeric_limits<double>::epsilon(), -0.25);
    const double slope = z / transverse;
    if (std::abs(slope) < expansion_threshold) {
      return std::log(slope + std::sqrt(slope * slope + 1.0));
    }
    return z > 0.0 ? std::log(2.0 * slope + 0.5 / slope) : -std::log(-2.0 * slope);
  }
  if (z == 0.0) return 0.0;
  return z > 0.0 ? z + eta_max : z - eta_max;
}

class P4 {
 public:
  P4() = default;

  P4(double pt, double eta, double phi, double mass)
      : pt_(pt), eta_(eta), phi_(phi), mass_(mass) {
    restrict_phi();
    restrict_negative_mass();
  }

  double Pt() const { return pt_; }
  double Eta() const { return eta_; }
  double Phi() const { return phi_; }
  double M() const { return mass_; }

  double Px() const { return pt_ * std::cos(phi_); }
  double Py() const { return pt_ * std::sin(phi_); }
  double Pz() const {
    if (pt_ > 0.0) return pt_ * std::sinh(eta_);
    if (eta_ == 0.0) return 0.0;
    return eta_ > 0.0 ? eta_ - eta_max : eta_ + eta_max;
  }

  double P() const {
    if (pt_ > 0.0) return pt_ * std::cosh(eta_);
    if (eta_ > eta_max) return eta_ - eta_max;
    return eta_ < -eta_max ? -eta_ - eta_max : 0.0;
  }
  double P2() const {
    const double p = P();
    return p * p;
  }
  double M2() const { return mass_ >= 0.0 ? mass_ * mass_ : -mass_ * mass_; }
  double E() const {
    const double e2 = P2() + M2();
    return std::sqrt(e2 > 0.0 ? e2 : 0.0);
  }

  double Theta() const {
    if (pt_ > 0.0) return 2.0 * std::atan(std::exp(-eta_));
    return eta_ >= 0.0 ? 0.0 : pi;
  }

  double Rapidity() const {
    const double e = E();
    const double z = Pz();
    return 0.5 * std::log((e + z) / (e - z));
  }

  // GenVector assigns from Cartesian coordinates here. Its assignment
  // intentionally differs from the cylindrical constructor's restrictions.
  P4& SetPxPyPzE(double x, double y, double z, double e) {
    pt_ = std::sqrt(x * x + y * y);
    eta_ = cylindrical_eta(pt_, z);
    phi_ = x == 0.0 && y == 0.0 ? 0.0 : std::atan2(y, x);
    const double mass2 = e * e - x * x - y * y - z * z;
    mass_ = mass2 >= 0.0 ? std::sqrt(mass2) : -std::sqrt(-mass2);
    return *this;
  }

  P4& operator*=(double factor) {
    if (factor < 0.0) {
      phi_ = phi_ > 0.0 ? phi_ - pi : phi_ + pi;
      eta_ = -eta_;
      factor = -factor;
    }
    pt_ *= factor;
    mass_ *= factor;
    return *this;
  }
  P4 operator*(double factor) const {
    P4 scaled = *this;
    return scaled *= factor;
  }

 private:
  void restrict_phi() {
    if (phi_ <= -pi || phi_ > pi) {
      phi_ -= std::floor(phi_ / (2.0 * pi) + 0.5) * 2.0 * pi;
    }
  }

  void restrict_negative_mass() {
    if (mass_ < 0.0 && P2() - mass_ * mass_ < 0.0) mass_ = -P();
  }

  double pt_ = 0.0;
  double eta_ = 0.0;
  double phi_ = 0.0;
  double mass_ = 0.0;
};

// The older Cartesian TLorentzVector path appears in nominal writeback and
// DeltaR. Its spatial eta formula is distinct from Cartesian GenVector.
class CartesianP4 {
 public:
  CartesianP4() = default;
  CartesianP4(double x, double y, double z, double e) : x_(x), y_(y), z_(z), energy_(e) {}

  double Px() const { return x_; }
  double Py() const { return y_; }
  double Pz() const { return z_; }
  double E() const { return energy_; }
  double Phi() const {
    if (x_ != 0.0) return std::atan2(y_, x_);
    if (y_ == 0.0) return 0.0;
    return y_ > 0.0 ? pi / 2.0 : -pi / 2.0;
  }
  double Eta() const {
    const double p = std::sqrt(x_ * x_ + y_ * y_ + z_ * z_);
    const double cosine = p == 0.0 ? 1.0 : z_ / p;
    if (cosine * cosine < 1.0) {
      return -0.5 * std::log((1.0 - cosine) / (1.0 + cosine));
    }
    if (z_ == 0.0) return 0.0;
    return z_ > 0.0 ? 1e11 : -1e11;
  }
  double Rapidity() const { return 0.5 * std::log((energy_ + z_) / (energy_ - z_)); }

  void SetPtEtaPhiM(double pt, double eta, double phi, double mass) {
    const double transverse = std::abs(pt);
    x_ = transverse * std::cos(phi);
    y_ = transverse * std::sin(phi);
    z_ = transverse * std::sinh(eta);
    if (mass >= 0.0) {
      energy_ = std::sqrt(x_ * x_ + y_ * y_ + z_ * z_ + mass * mass);
    } else {
      const double e2 = x_ * x_ + y_ * y_ + z_ * z_ - mass * mass;
      energy_ = std::sqrt(e2 >= 0.0 ? e2 : 0.0);
    }
  }

  double DeltaR(const CartesianP4& other, bool use_rapidity = false) const {
    const double longitudinal = use_rapidity ? Rapidity() - other.Rapidity() : Eta() - other.Eta();
    double phi = Phi() - other.Phi();
    while (phi >= pi) phi -= 2.0 * pi;
    while (phi < -pi) phi += 2.0 * pi;
    return std::sqrt(longitudinal * longitudinal + phi * phi);
  }

 private:
  double x_ = 0.0;
  double y_ = 0.0;
  double z_ = 0.0;
  double energy_ = 0.0;
};

}  // namespace jets::portable
