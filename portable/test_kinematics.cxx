#include "kinematics.h"
#include "random.h"

#include <Math/Vector4D.h>
#include <TError.h>
#include <TLorentzVector.h>
#include <TRandom3.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace {

using Reference = ROOT::Math::PtEtaPhiMVector;
using jets::portable::CartesianP4;
using jets::portable::P4;
constexpr double pi = jets::portable::pi;
std::size_t checks = 0;

void equal(double reference, double portable, std::string_view field) {
  ++checks;
  if (std::bit_cast<std::uint64_t>(reference) == std::bit_cast<std::uint64_t>(portable) ||
      (std::isnan(reference) && std::isnan(portable))) return;
  std::cerr << std::setprecision(17) << field << ": release " << reference << ", portable " << portable << '\n';
  throw std::runtime_error("Portable primitive differs from pinned ROOT");
}

void compare(const Reference& ref, const P4& candidate) {
  equal(ref.Pt(), candidate.Pt(), "cylindrical pt");
  equal(ref.Eta(), candidate.Eta(), "cylindrical eta");
  equal(ref.Phi(), candidate.Phi(), "cylindrical phi");
  equal(ref.M(), candidate.M(), "cylindrical mass");
  equal(ref.Px(), candidate.Px(), "cylindrical px");
  equal(ref.Py(), candidate.Py(), "cylindrical py");
  equal(ref.Pz(), candidate.Pz(), "cylindrical pz");
  equal(ref.P(), candidate.P(), "cylindrical p");
  equal(ref.P2(), candidate.P2(), "cylindrical p squared");
  equal(ref.M2(), candidate.M2(), "cylindrical mass squared");
  equal(ref.E(), candidate.E(), "cylindrical energy");
  equal(ref.Theta(), candidate.Theta(), "cylindrical theta");
  equal(ref.Rapidity(), candidate.Rapidity(), "cylindrical rapidity");
}

void compare(const TLorentzVector& ref, const CartesianP4& candidate) {
  equal(ref.Px(), candidate.Px(), "Cartesian px");
  equal(ref.Py(), candidate.Py(), "Cartesian py");
  equal(ref.Pz(), candidate.Pz(), "Cartesian pz");
  equal(ref.Eta(), candidate.Eta(), "Cartesian eta");
  equal(ref.Phi(), candidate.Phi(), "Cartesian phi");
  equal(ref.E(), candidate.E(), "Cartesian energy");
  equal(ref.Rapidity(), candidate.Rapidity(), "Cartesian rapidity");
}

void check_vectors() {
  const double near_pi = std::nextafter(pi, 0.0);
  const double beyond_pi = std::nextafter(pi, std::numeric_limits<double>::infinity());
  for (double pt : {-20.0, -0.0, 0.0, 0.01, 40'000.0}) {
    for (double eta : {-22758.0, -35.0, -2.3, -0.0, 0.0, 2.3, 35.0, 22758.0}) {
      for (double phi : {-17 * pi, -beyond_pi, -pi, -near_pi, -0.0, 0.0, 0.71, near_pi, pi, beyond_pi, 17 * pi}) {
        for (double mass : {-120.0, -2.0, -0.0, 0.0, 6.0}) {
          const Reference ref(pt, eta, phi, mass);
          const P4 candidate(pt, eta, phi, mass);
          compare(ref, candidate);
          for (double factor : {-2.25, -0.0, 0.0, 1.0, 1.25}) {
            compare(ref * factor, candidate * factor);
          }

          TLorentzVector cartesian_ref;
          CartesianP4 cartesian_candidate;
          cartesian_ref.SetPtEtaPhiM(pt, eta, phi, mass);
          cartesian_candidate.SetPtEtaPhiM(pt, eta, phi, mass);
          compare(cartesian_ref, cartesian_candidate);
        }
      }
    }
  }
  constexpr std::array coordinates{-37.0, -1.0, -0.0, 0.0, 0.001, 1.0, 41.0};
  for (double x : coordinates) {
    for (double y : coordinates) {
      for (double z : coordinates) {
        for (double energy : {0.0, 1.0, 50.0}) {
          Reference ref;
          P4 candidate;
          ref.SetPxPyPzE(x, y, z, energy);
          candidate.SetPxPyPzE(x, y, z, energy);
          compare(ref, candidate);
          const TLorentzVector spherical(x, y, z, energy);
          const CartesianP4 portable(x, y, z, energy);
          compare(spherical, portable);
          const TLorentzVector near_negative(-1.0, -1e-8, 2.0, 5.0);
          const CartesianP4 portable_negative(-1.0, -1e-8, 2.0, 5.0);
          equal(spherical.DeltaR(near_negative), portable.DeltaR(portable_negative), "delta R (pseudorapidity)");
          equal(spherical.DeltaR(near_negative, true), portable.DeltaR(portable_negative, true),
                "delta R (rapidity)");
        }
      }
    }
  }
}

void check_random() {
  constexpr std::array<std::uint64_t, 8> seeds{1, 12345, 54321, 65539, 314159, 0xffffffff, 0x100000000, 0x100000001};
  for (auto seed : seeds) {
    TRandom3 root_rng;
    root_rng.SetSeed(seed);
    jets::portable::Random portable_rng(seed);
    for (int index = 0; index < 8192; ++index) {
      equal(root_rng.Rndm(), portable_rng.Rndm(), "uniform MT19937 stream");
    }
    root_rng.SetSeed(seed);
    portable_rng.SetSeed(seed);
    for (int index = 0; index < 8192; ++index) {
      const double sigma = (index % 9) * 0.175;
      equal(root_rng.Gaus(1.0, sigma), portable_rng.Gaus(1.0, sigma), "Gaussian ACR stream");
      equal(root_rng.Rndm(), portable_rng.Rndm(), "next uniform after Gaussian draw");
    }
  }
}

}  // namespace

int main() {
  gErrorIgnoreLevel = kFatal;
  check_vectors();
  check_random();
  std::cout << checks << " exact portable four-vector and random-stream comparisons\n";
}
