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
 * @brief Reaction-field scheme
 */
class ReactionField : public EnergyImplementation<ReactionField> {
  private:
    double epsRF; //!< Relative permittivity of the surrounding medium
    double epsr;  //!< Relative permittivity of the dispersing medium
    bool shifted; //!< Shifted to zero potential at the cut-off
    const double pi = 4.0 * std::atan(1.0);

  public:
    /**
     * @param cutoff distance cutoff
     * @param epsRF relative dielectric constant of the surrounding
     * @param epsr relative dielectric constant of the sample
     * @param shifted shifted potential
     */
    inline ReactionField(double cutoff, double epsRF, double epsr, bool shifted)
        : EnergyImplementation(Scheme::reactionfield, cutoff), epsRF(epsRF), epsr(epsr), shifted(shifted) {
        name = "Reaction-field";
        has_dipolar_selfenergy = true;
        doi = "10.1080/00268977300102101";
        setSelfEnergyPrefactor({-3.0 * epsRF * double(shifted) / (4.0 * epsRF + 2.0 * epsr),
                                -(2.0 * epsRF - 2.0 * epsr) / (2.0 * (2.0 * epsRF + epsr))});
        setSelfFieldPrefactor({0.0, 0.0});
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
        chi = -6.0 * cutoff * cutoff * pi * ((-10.0 * double(shifted) / 3.0 + 4.0) * epsRF + epsr) /
              ((5.0 * (2.0 * epsRF + epsr)));
    }

    inline double short_range_function(double q) const override {
        return (1.0 + (epsRF - epsr) * q * q * q / (2.0 * epsRF + epsr) -
                3.0 * epsRF * q / (2.0 * epsRF + epsr) * double(shifted));
    }
    inline double short_range_function_derivative(double q) const override {
        return (3.0 * (epsRF - epsr) * q * q / (2.0 * epsRF + epsr) -
                3.0 * epsRF * double(shifted) / (2.0 * epsRF + epsr));
    }
    inline double short_range_function_second_derivative(double q) const override {
        return 6.0 * (epsRF - epsr) * q / (2.0 * epsRF + epsr);
    }
    inline double short_range_function_third_derivative(double) const override {
        return 6.0 * (epsRF - epsr) / (2.0 * epsRF + epsr);
    }

#ifdef NLOHMANN_JSON_HPP
    /** Construct using JSON object looking for the keywords `cutoff`, `epsRF`, `epsr`, and `shifted` */
    inline ReactionField(const nlohmann::json &j)
        : ReactionField(j.at("cutoff").get<double>(), j.at("epsRF").get<double>(), j.at("epsr").get<double>(),
                        j.at("shifted").get<bool>()) {}

  private:
    inline void _to_json(nlohmann::json &j) const override {
        j = {{"epsr", epsr}, {"epsRF", epsRF}, {"shifted", shifted}};
    }
#endif
};

} // namespace CoulombGalore
