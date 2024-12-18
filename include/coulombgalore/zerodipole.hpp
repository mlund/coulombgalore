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
 * @brief Zero-dipole scheme
 */
class ZeroDipole : public EnergyImplementation<ZeroDipole> {
  private:
    double alphaRed, alphaRed2; //!< Reduced damping-parameter, and squared
    const double pi_sqrt = 2.0 * std::sqrt(std::atan(1.0));
    const double pi = 4.0 * std::atan(1.0);

  public:
    /**
     * @param cutoff distance cutoff
     * @param alpha damping-parameter
     */
    inline ZeroDipole(double cutoff, double alpha) : EnergyImplementation(Scheme::zerodipole, cutoff) {
        name = "ZeroDipole";
        has_dipolar_selfenergy = true;
        doi = "10.1063/1.3582791";
        alphaRed = alpha * cutoff;
        alphaRed2 = alphaRed * alphaRed;
        setSelfEnergyPrefactor(
            {-alphaRed * (1.0 + 0.5 * std::exp(-alphaRed2)) / pi_sqrt - 0.75 * std::erfc(alphaRed),
             -alphaRed * (2.0 * alphaRed2 * (1.0 / 3.0) + std::exp(-alphaRed2)) / pi_sqrt - 0.5 * std::erfc(alphaRed)});
        setSelfFieldPrefactor({0.0, 0.0});
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
        chi = cutoff * cutoff *
              ((6.0 * alphaRed2 - 15.0) * std::erf(alphaRed) * pi - 6.0 * pi * alphaRed2 +
               (8.0 * alphaRed2 + 30.0) * alphaRed * std::exp(-alphaRed2) * std::sqrt(pi)) /
              (15.0 * alphaRed2);
    }

    inline double short_range_function(double q) const override {
        return (std::erfc(alphaRed * q) - q * std::erfc(alphaRed) +
                0.5 * (q * q - 1.0) * q * (std::erfc(alphaRed) + 2.0 * alphaRed * std::exp(-alphaRed2) / pi_sqrt));
    }
    inline double short_range_function_derivative(double q) const override {
        return (alphaRed * ((3.0 * q * q - 1.0) * std::exp(-alphaRed2) - 2.0 * std::exp(-alphaRed2 * q * q)) / pi_sqrt +
                1.5 * std::erfc(alphaRed) * (q * q - 1.0));
    }
    inline double short_range_function_second_derivative(double q) const override {
        return (2.0 * alphaRed * q * (2.0 * alphaRed2 * std::exp(-alphaRed2 * q * q) + 3.0 * std::exp(-alphaRed2)) /
                    pi_sqrt +
                3.0 * q * std::erfc(alphaRed));
    }
    inline double short_range_function_third_derivative(double q) const override {
        return (2.0 * alphaRed *
                    (2.0 * alphaRed2 * (1.0 - 2.0 * alphaRed2 * q * q) * std::exp(-alphaRed2 * q * q) +
                     3.0 * std::exp(-alphaRed2)) /
                    pi_sqrt +
                3.0 * std::erfc(alphaRed));
    }

#ifdef NLOHMANN_JSON_HPP
    /** Construct from JSON object, looking for `cutoff`, `alpha` */
    inline ZeroDipole(const nlohmann::json &j)
        : ZeroDipole(j.at("cutoff").get<double>(), j.at("alpha").get<double>()) {}

  private:
    inline void _to_json(nlohmann::json &j) const override { j = {{"alpha", alpha}}; }
#endif
};

} // namespace CoulombGalore
