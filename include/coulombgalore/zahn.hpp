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
#include "coulombgalore/core.hpp"

namespace CoulombGalore {
/**
 * @brief Zahn scheme
 */
class Zahn : public EnergyImplementation<Zahn> {
  private:
    double alphaRed, alphaRed2; //!< Reduced damping-parameter, and squared
    const double pi_sqrt = 2.0 * std::sqrt(std::atan(1.0));
    const double pi = 4.0 * std::atan(1.0);

  public:
    /**
     * @brief Contructor
     * @param cutoff distance cutoff
     * @param alpha damping-parameter
     */
    inline Zahn(double cutoff, double alpha) : EnergyImplementation(Scheme::zahn, cutoff) {
        name = "Zahn";
        has_dipolar_selfenergy = false;
        doi = "10.1021/jp025949h";
        alphaRed = alpha * cutoff;
        alphaRed2 = alphaRed * alphaRed;
        setSelfEnergyPrefactor({-alphaRed * (1.0 - std::exp(-alphaRed2)) / pi_sqrt + 0.5 * std::erfc(alphaRed),
                                0.0}); // Dipole self-energy undefined!
        setSelfFieldPrefactor({0.0, 0.0});
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
        chi = -(2.0 * (alphaRed * (alphaRed2 - 3.0) * std::exp(-alphaRed2) -
                       0.5 * std::sqrt(pi) * ((7.0 - 3.0 / alphaRed2) * std::erf(alphaRed) - 7.0) * alphaRed2)) *
              cutoff * cutoff * std::sqrt(pi) / (3.0 * alphaRed2);
    }

    inline double short_range_function(double q) const override {
        return (std::erfc(alphaRed * q) -
                (q - 1.0) * q * (std::erfc(alphaRed) + 2.0 * alphaRed * std::exp(-alphaRed2) / pi_sqrt));
    }
    inline double short_range_function_derivative(double q) const override {
        return (-(4.0 * (0.5 * std::exp(-alphaRed2 * q * q) * alphaRed +
                         (alphaRed * std::exp(-alphaRed2) + 0.5 * pi_sqrt * std::erfc(alphaRed)) * (q - 0.5))) /
                pi_sqrt);
    }
    inline double short_range_function_second_derivative(double q) const override {
        return (4.0 * (alphaRed2 * alphaRed * q * std::exp(-alphaRed2 * q * q) - alphaRed * std::exp(-alphaRed2) -
                       0.5 * pi_sqrt * std::erfc(alphaRed))) /
               pi_sqrt;
    }
    inline double short_range_function_third_derivative(double q) const override {
        return (-8.0 * std::exp(-alphaRed2 * q * q) * (alphaRed2 * q * q - 0.5) * alphaRed2 * alphaRed / pi_sqrt);
    }

#ifdef NLOHMANN_JSON_HPP
    /** Construct from JSON object, looking for keywords `cutoff`, `alpha` */
    inline Zahn(const nlohmann::json &j) : Zahn(j.at("cutoff").get<double>(), j.at("alpha").get<double>()) {}

  private:
    inline void _to_json(nlohmann::json &j) const override { j = {{"alpha", alpha}}; }
#endif
};

} // namespace CoulombGalore
