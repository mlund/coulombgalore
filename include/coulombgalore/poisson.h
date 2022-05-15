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
 * @brief Poisson scheme with and without specified Debye-length
 *
 * A general scheme which pending two parameters `C` and `D` can model several different pair-potentials.
 * The short-ranged function is
 *
 * @f[
 * S(q) = (1-\tilde{q})^{D+1}\sum_{c=0}^{C-1}\frac{C-c}{C}{D-1+c\choose c}\tilde{q}^c
 * @f]
 * where `C` is the number of cancelled derivatives at origin -2 (starting from second derivative),
 * and  `D` is the number of cancelled derivatives at the cut-off (starting from zeroth derivative)
 *
 * For infinite Debye-length the following holds:
 *
 * Type         | `C` | `D` | Reference / Comment
 * ------------ | --- | --- | ----------------------
 * `plain`      |  1  | -1  | Plain Coulomb
 * `wolf`       |  1  |  0  | Undamped Wolf, doi:10.1063/1.478738
 * `fennell`    |  1  |  1  | Levitt/undamped Fennell, doi:10/fp959p or 10/bqgmv2
 * `kale`       |  1  |  2  | Kale, doi:10/csh8bg
 * `mccann`     |  1  |  3  | McCann, doi:10.1021/ct300961
 * `fukuda`     |  2  |  1  | Undamped Fukuda, doi:10.1063/1.3582791
 * `markland`   |  2  |  2  | Markland, doi:10.1016/j.cplett.2008.09.019
 * `stenqvist`  |  3  |  3  | Stenqvist, doi:10/c5fr
 * `fanourgakis`|  4  |  3  | Fanourgakis, doi:10.1063/1.3216520
 *
 *  More info:
 *
 *  - http://dx.doi.org/10.1088/1367-2630/ab1ec1
 *
 * @warning Need to fix Yukawa-dipole self-energy
 */
class Poisson : public EnergyImplementation<Poisson> {
  private:
    signed int C;                 //!< Derivative cancelling-parameters
    signed int D;                 //!< Derivative cancelling-parameters
    double reduced_kappa;         //!< Reduced, inverse Debye length = cut-off / Debye length
    double reduced_kappa_squared; //!< Squared Reduced, inverse Debye length = (cut-off / Debye length)^2
    bool use_yukawa_screening;    //!< Use screening?
    double yukawa_denom;
    double binomCDC;

  public:
    /**
     * @param cutoff Spherical cutoff distance
     * @param C number of cancelled derivatives at origin -2 (starting from second derivative)
     * @param D number of cancelled derivatives at the cut-off (starting from zeroth derivative)
     * @param debye_length Debye screening length (infinite by default)
     */
    inline Poisson(double cutoff, signed int C, signed int D, double debye_length = infinity)
        : EnergyImplementation(Scheme::poisson, cutoff, debye_length), C(C), D(D) {
        if (C < 1)
            throw std::runtime_error("`C` must be larger than zero");
        if ((D < -1) && (D != -C))
            throw std::runtime_error("If `D` is less than negative one, then it has to equal negative `C`");
        if ((D == 0) && (C != 1))
            throw std::runtime_error("If `D` is zero, then `C` has to equal one ");
        name = "poisson";
        has_dipolar_selfenergy = true;
        if (C < 2)
            has_dipolar_selfenergy = false;
        doi = "10/c5fr";
        double a1 = -double(C + D) / double(C);
        reduced_kappa = 0.0;
        use_yukawa_screening = false;
        if (!std::isinf(debye_length)) {
            reduced_kappa = cutoff / debye_length;
            if (std::fabs(reduced_kappa) > 1e-6) {
                use_yukawa_screening = true;
                reduced_kappa_squared = reduced_kappa * reduced_kappa;
                yukawa_denom = 1.0 / (1.0 - std::exp(2.0 * reduced_kappa));
                a1 *= -2.0 * reduced_kappa * yukawa_denom;
            }
        }
        binomCDC = 0.0;
        if (D != -C) {
            binomCDC = double(binomial(C + D, C) * D);
        }
        setSelfEnergyPrefactor({0.5 * a1, 0.0}); // Dipole self-energy seems to be 0 for C >= 2
        setSelfFieldPrefactor({0.0, 0.0});
        if (C == 2)
            setSelfEnergyPrefactor({0.5 * a1, -double(D) * (double(D * D) + 3.0 * double(D) + 2.0) / 12.0});
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) +
             short_range_function(0.0); // Is this OK for Yukawa-interactions?
        chi = -2.0 * std::acos(-1.0) * cutoff * cutoff * (1.0 + double(C)) * (2.0 + double(C)) /
              (3.0 * double(D + 1 + C) *
               double(D + 2 + C)); // not confirmed, but have worked for all tested values of 'C' and 'D'
    }

    inline double short_range_function(double q) const override {
        if (D == -C) {
            return 1.0;
        }
        double tmp = 0;
        double qp = q;
        if (use_yukawa_screening) {
            qp = (1.0 - std::exp(2.0 * reduced_kappa * q)) * yukawa_denom;
        }
        if ((D == 0) && (C == 1)) {
            return (1.0 - qp);
        }
        for (signed int c = 0; c < C; c++) {
            tmp += double(binomial(D - 1 + c, c)) * double(C - c) / double(C) * powi(qp, c);
        }
        return powi(1.0 - qp, D + 1) * tmp;
    }

    inline double short_range_function_derivative(double q) const override {
        if (D == -C) {
            return 0.0;
        }
        if ((D == 0) && (C == 1)) {
            return 0.0;
        }
        double qp = q;
        double dqpdq = 1.0;
        if (use_yukawa_screening) {
            const auto exp2kq = std::exp(2.0 * reduced_kappa * q);
            qp = (1.0 - exp2kq) * yukawa_denom;
            dqpdq = -2.0 * reduced_kappa * exp2kq * yukawa_denom;
        }
        double tmp1 = 1.0;
        double tmp2 = 0.0;
        for (int c = 1; c < C; c++) {
            const auto factor = binomial(D - 1 + c, c) * (C - c) / static_cast<double>(C);
            tmp1 += factor * powi(qp, c);
            tmp2 += factor * c * powi(qp, c - 1);
        }
        const auto dSdqp = -(D + 1) * powi(1.0 - qp, D) * tmp1 + powi(1.0 - qp, D + 1) * tmp2;
        return dSdqp * dqpdq;
    }

    inline double short_range_function_second_derivative(double q) const override {
        if (D == -C) {
            return 0.0;
        }
        if ((D == 0) && (C == 1)) {
            return 0.0;
        }
        double qp = q;
        double dqpdq = 1.0;
        double d2qpdq2 = 0.0;
        double dSdqp = 0.0;
        if (use_yukawa_screening) {
            qp = (1.0 - std::exp(2.0 * reduced_kappa * q)) * yukawa_denom;
            dqpdq = -2.0 * reduced_kappa * std::exp(2.0 * reduced_kappa * q) * yukawa_denom;
            d2qpdq2 = -4.0 * reduced_kappa_squared * std::exp(2.0 * reduced_kappa * q) * yukawa_denom;
            double tmp1 = 1.0;
            double tmp2 = 0.0;
            for (signed int i = 1; i < C; i++) {
                const auto b = static_cast<double>(binomial(D - 1 + i, i) * (C - i));
                tmp1 += b / C * powi(qp, i);
                tmp2 += b * i / C * powi(qp, i - 1);
            }
            dSdqp = -(D + 1) * powi(1.0 - qp, D) * tmp1 + powi(1.0 - qp, D + 1) * tmp2;
        }
        const auto d2Sdqp2 = binomCDC * powi(1.0 - qp, D - 1) * powi(qp, C - 1);
        return (d2Sdqp2 * dqpdq * dqpdq + dSdqp * d2qpdq2);
    };

    inline double short_range_function_third_derivative(double q) const override {
        if (D == -C) {
            return 0.0;
        }
        if ((D == 0) && (C == 1)) {
            return 0.0;
        }
        double qp = q;
        double dqpdq = 1.0;
        double d2qpdq2 = 0.0;
        double d3qpdq3 = 0.0;
        double d2Sdqp2 = 0.0;
        double dSdqp = 0.0;
        if (use_yukawa_screening) {
            qp = (1.0 - std::exp(2.0 * reduced_kappa * q)) * yukawa_denom;
            dqpdq = -2.0 * reduced_kappa * std::exp(2.0 * reduced_kappa * q) * yukawa_denom;
            d2qpdq2 = -4.0 * reduced_kappa_squared * std::exp(2.0 * reduced_kappa * q) * yukawa_denom;
            d3qpdq3 = -8.0 * reduced_kappa_squared * reduced_kappa * std::exp(2.0 * reduced_kappa * q) * yukawa_denom;
            d2Sdqp2 = binomCDC * powi(1.0 - qp, D - 1) * powi(qp, C - 1);
            double tmp1 = 1.0;
            double tmp2 = 0.0;
            for (signed int c = 1; c < C; c++) {
                tmp1 += double(binomial(D - 1 + c, c)) * double(C - c) / double(C) * powi(qp, c);
                tmp2 += double(binomial(D - 1 + c, c)) * double(C - c) / double(C) * double(c) * powi(qp, c - 1);
            }
            dSdqp = -(D + 1) * powi(1.0 - qp, D) * tmp1 + powi(1.0 - qp, D + 1) * tmp2;
        }
        const double d3Sdqp3 =
            binomCDC * powi(1.0 - qp, D - 2) * powi(qp, C - 2) * ((2.0 - double(C + D)) * qp + double(C) - 1.0);
        return (d3Sdqp3 * dqpdq * dqpdq * dqpdq + 3.0 * d2Sdqp2 * dqpdq * d2qpdq2 + dSdqp * d3qpdq3);
    };

#ifdef NLOHMANN_JSON_HPP
    /** Construct from JSON object, looking for keywords `cutoff`, `debyelength` (infinite), and coefficients `C` and
     * `D` */
    inline Poisson(const nlohmann::json &j)
        : Poisson(j.at("cutoff").get<double>(), j.at("C").get<int>(), j.at("D").get<int>(),
                  j.value("debyelength", infinity)) {}

  private:
    inline void _to_json(nlohmann::json &j) const override { j = {{"C", C}, {"D", D}}; }
#endif
};

} // namespace CoulombGalore