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
 * @brief Fanourgakis scheme.
 * @note This is the same as using the 'Poisson' approach with parameters 'C=4' and 'D=3'
 */
class Fanourgakis : public EnergyImplementation<Fanourgakis> {
  public:
    /**
     * @param cutoff distance cutoff
     */
    inline Fanourgakis(double cutoff) : EnergyImplementation(Scheme::fanourgakis, cutoff) {
        name = "fanourgakis";
        has_dipolar_selfenergy = true;
        doi = "10.1063/1.3216520";
        setSelfEnergyPrefactor({-0.875, 0.0});
        setSelfFieldPrefactor({0.0, 0.0});
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
        chi = -5.0 * std::acos(-1.0) * cutoff * cutoff / 18.0;
    }

    inline double short_range_function(double q) const override {
        const auto q2 = q * q;
        return powi(1.0 - q, 4) * (1.0 + 2.25 * q + 3.0 * q2 + 2.5 * q2 * q);
    }
    inline double short_range_function_derivative(double q) const override {
        return (-1.75 + 26.25 * powi(q, 4) - 42.0 * powi(q, 5) + 17.5 * powi(q, 6));
    }
    inline double short_range_function_second_derivative(double q) const override {
        return 105.0 * powi(q, 3) * powi(q - 1.0, 2);
    };
    inline double short_range_function_third_derivative(double q) const override {
        return 525.0 * powi(q, 2) * (q - 0.6) * (q - 1.0);
    };
#ifdef NLOHMANN_JSON_HPP
    /** Construct from JSON looking for keyword `cutoff` */
    inline Fanourgakis(const nlohmann::json &j) : Fanourgakis(j.at("cutoff").get<double>()) {}

  private:
    inline void _to_json(nlohmann::json &) const override {}
#endif
};


} // namespace CoulombGalore
