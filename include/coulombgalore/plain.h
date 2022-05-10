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
#include "core.h"

namespace CoulombGalore {

/**
 * @brief No truncation scheme, cutoff = infinity
 * @warning Neutralization-scheme should not be used using an infinite cut-off
 */
class Plain : public EnergyImplementation<Plain> {
  public:
    inline Plain(double debye_length = infinity)
        : EnergyImplementation(Scheme::plain, std::sqrt(std::numeric_limits<double>::max()), debye_length) {
        name = "plain";
        has_dipolar_selfenergy = true;
        doi = "Premier mémoire sur l’électricité et le magnétisme by Charles-Augustin de Coulomb"; // :P
        setSelfEnergyPrefactor({0.0, 0.0});
        setSelfFieldPrefactor({0.0, 0.0});
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
        chi = -2.0 * std::acos(-1.0) * cutoff_squared; // should not be used!
    };
    inline double short_range_function(double) const override { return 1.0; };
    inline double short_range_function_derivative(double) const override { return 0.0; }
    inline double short_range_function_second_derivative(double) const override { return 0.0; }
    inline double short_range_function_third_derivative(double) const override { return 0.0; }
#ifdef NLOHMANN_JSON_HPP
    inline Plain(const nlohmann::json &j) : Plain(j.value("debyelength", infinity)) {}

  private:
    inline void _to_json(nlohmann::json &) const override {}
#endif
};

} // namespace CoulombGalore