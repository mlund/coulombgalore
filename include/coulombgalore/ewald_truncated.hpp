/*
* CoulombGalore - A Library for Electrostatic Interactions
*
* MIT License
* Copyright (c) 2019 Björn Stenqvist and Mikael Lund
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
#pragma once

#include "ewald.h"

namespace CoulombGalore {

/**
 * @brief Ewald real-space scheme using a truncated Gaussian screening-function.
 */
class EwaldTruncated : public EnergyImplementation<EwaldTruncated> {
    double eta, eta2, eta3; //!< Reduced damping-parameter, and squared, and cubed
    // double zeta, zeta2, zeta3;             //!< Reduced inverse Debye-length, and squared, and cubed
    double surface_dielectric_constant; //!< Dielectric constant of the surrounding medium
    double F0;                          //!< 'scaling' of short-ranged function
    const double pi_sqrt = 2.0 * std::sqrt(std::atan(1.0));
    const double pi = 4.0 * std::atan(1.0);

  public:
    /**
     * @param cutoff distance cutoff
     * @param alpha damping-parameter
     */
    inline EwaldTruncated(double cutoff, double alpha, double surface_dielectric_constant = infinity,
                          [[maybe_unused]] double debye_length = infinity)
        : EnergyImplementation(Scheme::ewaldt, cutoff), surface_dielectric_constant(surface_dielectric_constant) {
        name = "EwaldT real-space";
        has_dipolar_selfenergy = true;
        doi = "XYZ";
        eta = alpha * cutoff;
        eta2 = eta * eta;
        eta3 = eta2 * eta;
        if (surface_dielectric_constant < 1.0) {
            surface_dielectric_constant = infinity;
        }
        F0 = 1.0 - std::erfc(eta) - 2.0 * eta / pi_sqrt * std::exp(-eta2);
        if (std::isinf(surface_dielectric_constant)) {
            T0 = 1.0;
        } else {
            T0 = 2.0 * (surface_dielectric_constant - 1.0) / (2.0 * surface_dielectric_constant + 1.0);
        }
        chi = -(1.0 - 4.0 * eta3 * std::exp(-eta2) / (3.0 * pi_sqrt * F0)) * cutoff_squared * pi / eta2;
        setSelfEnergyPrefactor({-eta / pi_sqrt * (1.0 - std::exp(-eta2)) / F0,
                                -eta3 * 2.0 / 3.0 / (std::erf(eta) * pi_sqrt - 2.0 * eta * std::exp(-eta2))});
        setSelfFieldPrefactor({0.0, 0.0}); // FIX
    }

    inline double short_range_function(double q) const override {
        return (std::erfc(eta * q) - std::erfc(eta) - (1.0 - q) * 2.0 * eta / pi_sqrt * std::exp(-eta2)) / F0;
    }
    inline double short_range_function_derivative(double q) const override {
        return -2.0 * eta * (std::exp(-eta2 * q * q) - std::exp(-eta2)) / pi_sqrt / F0;
    }
    inline double short_range_function_second_derivative(double q) const override {
        return 4.0 * eta3 * q * std::exp(-eta2 * q * q) / pi_sqrt / F0;
    }
    inline double short_range_function_third_derivative(double q) const override {
        return -8.0 * (eta2 * q * q - 0.5) * eta3 * std::exp(-eta2 * q * q) / pi_sqrt / F0;
    }

    /**
     * @brief Surface-term
     * @param positions Positions of particles
     * @param charges Charges of particles
     * @param dipoles Dipole moments of particles
     * @param volume Volume of unit-cell
     */
    inline double surface_energy(const std::vector<vec3> &positions, const std::vector<double> &charges,
                                 const std::vector<vec3> &dipoles, double volume) const {
        assert(positions.size() == charges.size());
        assert(positions.size() == dipoles.size());
        vec3 pos_times_charge_sum = {0.0, 0.0, 0.0};
        vec3 dipole_sum = {0.0, 0.0, 0.0};
        for (size_t i = 0; i < positions.size(); i++) {
            pos_times_charge_sum += positions[i] * charges[i];
            dipole_sum += dipoles[i];
        }
        const auto sqDipoles = pos_times_charge_sum.dot(pos_times_charge_sum) +
                               2.0 * pos_times_charge_sum.dot(dipole_sum) + dipole_sum.dot(dipole_sum);
        return (2.0 * pi / (2.0 * surface_dielectric_constant + 1.0) / volume * sqDipoles);
    }

#ifdef NLOHMANN_JSON_HPP
    inline EwaldTruncated(const nlohmann::json &j)
        : EwaldTruncated(j.at("cutoff").get<double>(), j.at("alpha").get<double>(), j.value("epss", infinity),
                         j.value("debyelength", infinity)) {}

  private:
    inline void _to_json(nlohmann::json &j) const override {
        j = {{"alpha", eta / cutoff}};
        if (std::isinf(surface_dielectric_constant)) {
            j["epss"] = "inf";
        } else {
            j["epss"] = surface_dielectric_constant;
        }
    }
#endif
};
} // namespace CoulombGalore
