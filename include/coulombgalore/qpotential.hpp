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
 * @brief Help-function for the q-potential scheme
 * @returns q-Pochhammer Symbol
 * @param q Normalized distance, q = r / Rcutoff
 * @param l Type of base interaction, l=0 is ion-ion, l=1 is ion-dipole, l=2 is dipole-dipole etc.
 * @param P Number of higher order moments to cancel
 *
 * @details The parameters are explaind in term of electrostatic moment cancellation as used in the q-potential scheme.
 * @f[
 *     (a;q)_P = \prod_{n=1}^P(1-aq^{n-1})
 * @f]
 * where @f$ a=q^l @f$. In the implementation we use that
 * @f$
 *     (q^l;q)_P = (1-q)^P\prod_{n=1}^P\sum_{k=0}^{n+l}q^k
 * @f$
 * which gives simpler expressions for the derivatives.
 *
 * More information here: http://mathworld.wolfram.com/q-PochhammerSymbol.html
 */
inline double qPochhammerSymbol(double q, int l = 0, int P = 300) {
    double Ct = 1.0; // evaluates to \prod_{n=1}^P\sum_{k=0}^{n+l}q^k
    for (int n = 1; n < P + 1; n++) {
        double val = 0.0;
        for (int k = 1; k < n + l + 1; k++) {
            val += powi(q, k - 1);
        }
        Ct *= val;
    }
    const auto Dt = powi(1.0 - q, P); // (1-q)^P
    return (Ct * Dt);
}

/**
 * @brief Gives the derivative of the q-Pochhammer Symbol
 */
inline double qPochhammerSymbolDerivative(double q, int l = 0, int P = 300) {
    double Ct = 1.0; // evaluates to \prod_{n=1}^P\sum_{k=0}^{n+l}q^k
    for (int n = 1; n < P + 1; n++) {
        double val = 0.0;
        for (int k = 1; k < n + l + 1; k++) {
            val += powi(q, k - 1);
        }
        Ct *= val;
    }
    double dCt = 0.0; // evaluates to derivative of \prod_{n=1}^P\sum_{k=0}^{n+l}q^k
    for (int n = 1; n < P + 1; n++) {
        double nom = 0.0;
        double denom = 1.0;
        for (int k = 2; k < n + l + 1; k++) {
            nom += (k - 1) * powi(q, k - 2);
            denom += powi(q, k - 1);
        }
        dCt += nom / denom;
    }
    dCt *= Ct;
    double Dt = powi(1.0 - q, P); // (1-q)^P
    double dDt = 0.0;
    if (P > 0) {
        dDt = -P * powi(1 - q, P - 1);
    } // derivative of (1-q)^P
    return (Ct * dDt + dCt * Dt);
}

/**
 * @brief Gives the second derivative of the q-Pochhammer Symbol
 */
inline double qPochhammerSymbolSecondDerivative(double q, int l = 0, int P = 300) {
    double Ct = 1.0; // evaluates to \prod_{n=1}^P\sum_{k=0}^{n+l}q^k
    double DS = 0.0;
    double dDS = 0.0;
    for (int n = 1; n < P + 1; n++) {
        double tmp = 0.0;
        for (int k = 1; k < n + l + 1; k++)
            tmp += powi(q, k - 1);
        Ct *= tmp;
        double nom = 0.0;
        double denom = 1.0;
        for (int k = 2; k < n + l + 1; k++) {
            nom += (k - 1) * powi(q, k - 2);
            denom += powi(q, k - 1);
        }
        DS += nom / denom;
        double diffNom = 0.0;
        double diffDenom = 1.0;
        for (int k = 3; k < n + l + 1; k++) {
            diffNom += (k - 1) * (k - 2) * powi(q, k - 3);
            diffDenom += (k - 1) * powi(q, k - 2);
        }
        dDS += (diffNom * denom - nom * diffDenom) / denom / denom;
    }
    double dCt = Ct * DS;              // derivative of \prod_{n=1}^P\sum_{k=0}^{n+l}q^k
    double ddCt = dCt * DS + Ct * dDS; // second derivative of \prod_{n=1}^P\sum_{k=0}^{n+l}q^k
    double Dt = powi(1.0 - q, P);      // (1-q)^P
    double dDt = 0.0;
    if (P > 0) {
        dDt = -P * powi(1 - q, P - 1);
    } // derivative of (1-q)^P
    double ddDt = 0.0;
    if (P > 1) {
        ddDt = P * (P - 1) * powi(1 - q, P - 2);
    } // second derivative of (1-q)^P
    return (Ct * ddDt + 2 * dCt * dDt + ddCt * Dt);
}

/**
 * @brief Gives the third derivative of the q-Pochhammer Symbol
 */
inline double qPochhammerSymbolThirdDerivative(double q, int l = 0, int P = 300) {
    double Ct = 1.0; // evaluates to \prod_{n=1}^P\sum_{k=0}^{n+l}q^k
    double DS = 0.0;
    double dDS = 0.0;
    double ddDS = 0.0;
    for (int n = 1; n < P + 1; n++) {
        double tmp = 0.0;
        for (int k = 1; k < n + l + 1; k++)
            tmp += powi(q, k - 1);
        Ct *= tmp;
        double f = 0.0;
        double g = 1.0;
        for (int k = 2; k < n + l + 1; k++) {
            f += (k - 1) * powi(q, k - 2);
            g += powi(q, k - 1);
        }
        DS += f / g;
        double df = 0.0;
        double dg = 0.0;
        if (n + l > 1)
            dg = 1.0;
        for (int k = 3; k < n + l + 1; k++) {
            df += (k - 1) * (k - 2) * powi(q, k - 3);
            dg += (k - 1) * powi(q, k - 2);
        }
        dDS += (df * g - f * dg) / g / g;
        double ddf = 0.0;
        double ddg = 0.0;
        if (n + l > 2) {
            ddg = 2.0;
        }
        for (int k = 4; k < n + l + 1; k++) {
            ddf += (k - 1) * (k - 2) * (k - 3) * powi(q, k - 4);
            ddg += (k - 1) * (k - 2) * powi(q, k - 3);
        }
        ddDS += (ddf * g * g - 2.0 * df * dg * g + 2.0 * f * dg * dg - f * ddg * g) / g / g / g;
    }
    double dCt = Ct * DS;                                   // derivative of \prod_{n=1}^P\sum_{k=0}^{n+l}q^k
    double ddCt = dCt * DS + Ct * dDS;                      // second derivative of \prod_{n=1}^P\sum_{k=0}^{n+l}q^k
    double dddCt = ddCt * DS + 2.0 * dCt * dDS + Ct * ddDS; // third derivative of \prod_{n=1}^P\sum_{k=0}^{n+l}q^k
    double Dt = powi(1.0 - q, P);                           // (1-q)^P
    double dDt = 0.0;
    if (P > 0) {
        dDt = -P * powi(1 - q, P - 1);
    } // derivative of (1-q)^P
    double ddDt = 0.0;
    if (P > 1) {
        ddDt = P * (P - 1) * powi(1 - q, P - 2);
    } // second derivative of (1-q)^P
    double dddDt = 0.0;
    if (P > 2) {
        dddDt = -P * (P - 1) * (P - 2) * powi(1 - q, P - 3);
    } // third derivative of (1-q)^P
    return (dddCt * Dt + 3.0 * ddCt * dDt + 3 * dCt * ddDt + Ct * dddDt);
}

// -------------- qPotential ---------------
template <int order> class qPotentialFixedOrder : public EnergyImplementation<qPotentialFixedOrder<order>> {
  public:
    typedef EnergyImplementation<qPotentialFixedOrder<order>> base;
    using base::chi;
    using base::name;
    using base::T0;
    /**
     * @param cutoff distance cutoff
     * @param order number of moments to cancel
     */
    inline qPotentialFixedOrder(double cutoff) : base(Scheme::qpotential, cutoff) {
        name = "qpotential";
        this->has_dipolar_selfenergy = true;
        this->doi = "10.1039/c9cp03875b";
        this->setSelfEnergyPrefactor({-0.5, -0.5});
        this->setSelfFieldPrefactor({0.0, 0.0});
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
        chi = -86459.0 * std::acos(-1.0) * cutoff * cutoff / 235620.0;
    }

    inline double short_range_function(double q) const override { return qPochhammerSymbol(q, 0, order); }
    inline double short_range_function_derivative(double q) const override {
        return qPochhammerSymbolDerivative(q, 0, order);
    }
    inline double short_range_function_second_derivative(double q) const override {
        return qPochhammerSymbolSecondDerivative(q, 0, order);
    }
    inline double short_range_function_third_derivative(double q) const override {
        return qPochhammerSymbolThirdDerivative(q, 0, order);
    }

#ifdef NLOHMANN_JSON_HPP
    inline qPotentialFixedOrder(const nlohmann::json &j) : qPotentialFixedOrder(j.at("cutoff").get<double>()) {}

  private:
    inline void _to_json(nlohmann::json &j) const override { j = {{"order", order}}; }
#endif
};

/**
 * @brief qPotential scheme
 * @note http://dx.doi.org/10/c5fr
 *
 * The short-ranged function is
 *
 * @f[
 * S(q) = \prod_{n=1}^{\text{order}}(1-q^n)
 * @f]
 */
class qPotential : public EnergyImplementation<qPotential> {
  private:
    int order; //!< Number of moments to cancel

  public:
    /**
     * @param cutoff distance cutoff
     * @param order number of moments to cancel
     */
    inline qPotential(double cutoff, int order) : EnergyImplementation(Scheme::qpotential, cutoff), order(order) {
        name = "qpotential";
        has_dipolar_selfenergy = true;
        doi = "10.1039/c9cp03875b";
        setSelfEnergyPrefactor({-0.5, -0.5});
        setSelfFieldPrefactor({0.0, 0.0});
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
        chi = 0.0; // -Pi*Rc^2 * [  2/3   7/15  17/42 146/385  86459/235620 ]
    }

    inline double short_range_function(double q) const override { return qPochhammerSymbol(q, 0, order); }
    inline double short_range_function_derivative(double q) const override {
        return qPochhammerSymbolDerivative(q, 0, order);
    }
    inline double short_range_function_second_derivative(double q) const override {
        return qPochhammerSymbolSecondDerivative(q, 0, order);
    }
    inline double short_range_function_third_derivative(double q) const override {
        return qPochhammerSymbolThirdDerivative(q, 0, order);
    }

#ifdef NLOHMANN_JSON_HPP
    /** Construct from JSON object, looking for `cutoff`, `order` */
    inline qPotential(const nlohmann::json &j)
        : qPotential(j.at("cutoff").get<double>(), j.at("order").get<double>()) {}

  private:
    inline void _to_json(nlohmann::json &j) const override { j = {{"order", order}}; }
#endif
};
} // namespace CoulombGalore
