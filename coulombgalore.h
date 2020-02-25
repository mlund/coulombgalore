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

#include <string>
#include <limits>
#include <cmath>
#include <iostream>
#include <vector>
#include <array>
#include <functional>
#include <memory>
#include <Eigen/Core>
#include "Faddeeva.hh"

/** modern json for c++ added "_" suffix at around ~version 3.6 */
#ifdef NLOHMANN_JSON_HPP_
#define NLOHMANN_JSON_HPP
#endif

/** Namespace containing all of CoulombGalore */
namespace CoulombGalore {

/** Typedef for 3D vector such a position or dipole moment */
typedef Eigen::Vector3d vec3;
typedef Eigen::Matrix3d mat33;

constexpr double infinity = std::numeric_limits<double>::infinity(); //!< Numerical infinity

/** Enum defining all possible schemes */
enum class Scheme {
    plain,
    ewald,
    ewaldt,
    reactionfield,
    wolf,
    poisson,
    qpotential,
    fanourgakis,
    zerodipole,
    zahn,
    fennell,
    qpotential5,
    spline
};

/**
 * @brief n'th integer power of float
 *
 * On GCC/Clang this will use the fast `__builtin_powi` function.
 */
inline double powi(double x, int n) {
#if defined(__GNUG__)
    return __builtin_powi(x, n);
#else
    return std::pow(x, n);
#endif
}

/**
 * @brief Returns the factorial of 'n'. Note that 'n' must be positive semidefinite.
 * @note Calculated at compile time and thus have no run-time overhead.
 */
inline constexpr unsigned int factorial(unsigned int n) { return n <= 1 ? 1 : n * factorial(n - 1); }

constexpr unsigned int binomial(signed int n, signed int k) {
    return factorial(n) / (factorial(k) * factorial(n - k));
}

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
        for (int k = 1; k < n + l + 1; k++)
            val += powi(q, k - 1);
        Ct *= val;
    }
    double Dt = powi(1.0 - q, P); // (1-q)^P
    return (Ct * Dt);
}

/**
 * @brief Gives the derivative of the q-Pochhammer Symbol
 */
inline double qPochhammerSymbolDerivative(double q, int l = 0, int P = 300) {
    double Ct = 1.0; // evaluates to \prod_{n=1}^P\sum_{k=0}^{n+l}q^k
    for (int n = 1; n < P + 1; n++) {
        double val = 0.0;
        for (int k = 1; k < n + l + 1; k++)
            val += powi(q, k - 1);
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
    if (P > 0)
        dDt = -P * powi(1 - q, P - 1); // derivative of (1-q)^P
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
    if (P > 0)
        dDt = -P * powi(1 - q, P - 1); // derivative of (1-q)^P
    double ddDt = 0.0;
    if (P > 1)
        ddDt = P * (P - 1) * powi(1 - q, P - 2); // second derivative of (1-q)^P
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
        if (n + l > 2)
            ddg = 2.0;
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
    if (P > 0)
        dDt = -P * powi(1 - q, P - 1); // derivative of (1-q)^P
    double ddDt = 0.0;
    if (P > 1)
        ddDt = P * (P - 1) * powi(1 - q, P - 2); // second derivative of (1-q)^P
    double dddDt = 0.0;
    if (P > 2)
        dddDt = -P * (P - 1) * (P - 2) * powi(1 - q, P - 3); // third derivative of (1-q)^P
    return (dddCt * Dt + 3.0 * ddCt * dDt + 3 * dCt * ddDt + Ct * dddDt);
}

namespace Tabulate {

/* base class for all tabulators - no dependencies */
template <typename T = double> class TabulatorBase {
  protected:
    T utol = 1e-5, ftol = -1, umaxtol = -1, fmaxtol = -1;
    T numdr = 0.0001; // dr for derivative evaluation

    // First derivative with respect to x
    T f1(std::function<T(T)> f, T x) const { return (f(x + numdr * 0.5) - f(x - numdr * 0.5)) / (numdr); }

    // Second derivative with respect to x
    T f2(std::function<T(T)> f, T x) const { return (f1(f, x + numdr * 0.5) - f1(f, x - numdr * 0.5)) / (numdr); }

    void check() const {
        if (ftol != -1 && ftol <= 0.0) {
            std::cerr << "ftol=" << ftol << " too small\n" << std::endl;
            abort();
        }
        if (umaxtol != -1 && umaxtol <= 0.0) {
            std::cerr << "umaxtol=" << umaxtol << " too small\n" << std::endl;
            abort();
        }
        if (fmaxtol != -1 && fmaxtol <= 0.0) {
            std::cerr << "fmaxtol=" << fmaxtol << " too small\n" << std::endl;
            abort();
        }
    }

  public:
    struct data {
        std::vector<T> r2;      // r2 for intervals
        std::vector<T> c;       // c for coefficents
        T rmin2 = 0, rmax2 = 0; // useful to save these with table
        bool empty() const { return r2.empty() && c.empty(); }
        inline size_t numKnots() const { return r2.size(); }
    };

    void setTolerance(T _utol, T _ftol = -1, T _umaxtol = -1, T _fmaxtol = -1) {
        utol = _utol;
        ftol = _ftol;
        umaxtol = _umaxtol;
        fmaxtol = _fmaxtol;
    }

    void setNumdr(T _numdr) { numdr = _numdr; }
};

/*
 * @brief Andrea table with logarithmic search
 *
 * Tabulator with logarithmic search.
 * Code mainly from MolSim (Per Linse) with some upgrades
 * Reference: doi:10/frzp4d
 *
 * @note Slow on Intel compiler
 */
template <typename T = double> class Andrea : public TabulatorBase<T> {
  private:
    typedef TabulatorBase<T> base; // for convenience
    int mngrid = 1200;             // Max number of controlpoints
    int ndr = 100;                 // Max number of trials to decr dr
    T drfrac = 0.9;                // Multiplicative factor to decr dr

    std::vector<T> SetUBuffer(T, T zlow, T, T zupp, T u0low, T u1low, T u2low, T u0upp, T u1upp, T u2upp) {

        // Zero potential and force return no coefficients
        if (std::fabs(u0low) < 1e-9)
            if (std::fabs(u1low) < 1e-9)
                return {0, 0, 0, 0, 0, 0, 0};

        T dz1 = zupp - zlow;
        T dz2 = dz1 * dz1;
        T dz3 = dz2 * dz1;
        T w0low = u0low;
        T w1low = u1low;
        T w2low = u2low;
        T w0upp = u0upp;
        T w1upp = u1upp;
        T w2upp = u2upp;
        T c0 = w0low;
        T c1 = w1low;
        T c2 = w2low * 0.5;
        T a = 6 * (w0upp - c0 - c1 * dz1 - c2 * dz2) / dz3;
        T b = 2 * (w1upp - c1 - 2 * c2 * dz1) / dz2;
        T c = (w2upp - 2 * c2) / dz1;
        T c3 = (10 * a - 12 * b + 3 * c) / 6;
        T c4 = (-15 * a + 21 * b - 6 * c) / (6 * dz1);
        T c5 = (2 * a - 3 * b + c) / (2 * dz2);

        return {zlow, c0, c1, c2, c3, c4, c5};
    }

    /*
     * @returns boolean vector.
     * - `[0]==true`: tolerance is approved,
     * - `[1]==true` Repulsive part is found.
     */
    std::vector<bool> CheckUBuffer(std::vector<T> &ubuft, T rlow, T rupp, std::function<T(T)> f) const {

        // Number of points to control
        int ncheck = 11;
        T dr = (rupp - rlow) / (ncheck - 1);
        std::vector<bool> vb(2, false);

        for (int i = 0; i < ncheck; i++) {
            T r1 = rlow + dr * ((T)i);
            T r2 = r1 * r1;
            T u0 = f(r2);
            T u1 = base::f1(f, r2);
            T dz = r2 - rlow * rlow;
            T usum =
                ubuft.at(1) +
                dz * (ubuft.at(2) + dz * (ubuft.at(3) + dz * (ubuft.at(4) + dz * (ubuft.at(5) + dz * ubuft.at(6)))));

            T fsum = ubuft.at(2) +
                     dz * (2 * ubuft.at(3) + dz * (3 * ubuft.at(4) + dz * (4 * ubuft.at(5) + dz * (5 * ubuft.at(6)))));

            if (std::fabs(usum - u0) > base::utol)
                return vb;
            if (base::ftol != -1 && std::fabs(fsum - u1) > base::ftol)
                return vb;
            if (base::umaxtol != -1 && std::fabs(usum) > base::umaxtol)
                vb[1] = true;
            if (base::fmaxtol != -1 && std::fabs(usum) > base::fmaxtol)
                vb[1] = true;
        }
        vb[0] = true;
        return vb;
    }

  public:
    /*
     * @brief Get tabulated value at f(x)
     * @param d Table data
     * @param r2 value
     */
    inline T eval(const typename base::data &d, T r2) const {
        assert(r2!=0); // r2 cannot be *exactly* zero
        size_t pos = std::lower_bound(d.r2.begin(), d.r2.end(), r2) - d.r2.begin() - 1;
        size_t pos6 = 6 * pos;
        assert((pos6 + 5) < d.c.size() && "out of bounds");
        T dz = r2 - d.r2[pos];
        return d.c[pos6] +
               dz * (d.c[pos6 + 1] +
                     dz * (d.c[pos6 + 2] + dz * (d.c[pos6 + 3] + dz * (d.c[pos6 + 4] + dz * (d.c[pos6 + 5])))));
    }

    /*
     * @brief Get tabulated value at df(x)/dx
     * @param d Table data
     * @param r2 value
     */
    T evalDer(const typename base::data &d, T r2) const {
        size_t pos = std::lower_bound(d.r2.begin(), d.r2.end(), r2) - d.r2.begin() - 1;
        size_t pos6 = 6 * pos;
        T dz = r2 - d.r2[pos];
        return (d.c[pos6 + 1] +
                dz * (2.0 * d.c[pos6 + 2] +
                      dz * (3.0 * d.c[pos6 + 3] + dz * (4.0 * d.c[pos6 + 4] + dz * (5.0 * d.c[pos6 + 5])))));
    }

    /**
     * @brief Tabulate f(x) in interval ]min,max]
     */
    typename base::data generate(std::function<T(T)> f, double rmin, double rmax) {
        rmin = std::sqrt(rmin);
        rmax = std::sqrt(rmax);
        base::check();
        typename base::data td;
        td.rmin2 = rmin * rmin;
        td.rmax2 = rmax * rmax;

        T rumin = rmin;
        T rmax2 = rmax * rmax;
        T dr = rmax - rmin;
        T rupp = rmax;
        T zupp = rmax2;
        bool repul = false; // Stop tabulation if repul is true

        td.r2.push_back(zupp);

        int i;
        for (i = 0; i < mngrid; i++) {
            T rlow = rupp;
            T zlow;
            std::vector<T> ubuft;
            int j;

            dr = (rupp - rmin);

            for (j = 0; j < ndr; j++) {
                zupp = rupp * rupp;
                rlow = rupp - dr;
                if (rumin > rlow)
                    rlow = rumin;

                zlow = rlow * rlow;

                T u0low = f(zlow);
                T u1low = base::f1(f, zlow);
                T u2low = base::f2(f, zlow);
                T u0upp = f(zupp);
                T u1upp = base::f1(f, zupp);
                T u2upp = base::f2(f, zupp);

                ubuft = SetUBuffer(rlow, zlow, rupp, zupp, u0low, u1low, u2low, u0upp, u1upp, u2upp);
                std::vector<bool> vb = CheckUBuffer(ubuft, rlow, rupp, f);
                repul = vb[1];
                if (vb[0]) {
                    rupp = rlow;
                    break;
                }
                dr *= drfrac;
            }

            if (j >= ndr)
                throw std::runtime_error("Andrea spline: try to increase utol/ftol");
            if (ubuft.size() != 7)
                throw std::runtime_error("Andrea spline: wrong size of ubuft, min value + 6 coefficients");

            td.r2.push_back(zlow);
            for (size_t k = 1; k < ubuft.size(); k++)
                td.c.push_back(ubuft.at(k));

            // Entered a highly repulsive part, stop tabulation
            if (repul) {
                rumin = rlow;
                td.rmin2 = rlow * rlow;
            }
            if (rlow <= rumin || repul)
                break;
        }

        if (i >= mngrid)
            throw std::runtime_error("Andrea spline: try to increase utol/ftol");

            // create final reversed c and r2
#if __cplusplus >= 201703L
        // C++17 only code
        assert(td.c.size() % 6 == 0);
        assert(td.c.size() / (td.r2.size() - 1) == 6);
        assert(std::is_sorted(td.r2.rbegin(), td.r2.rend()));
        std::reverse(td.r2.begin(), td.r2.end());       // reverse all elements
        for (size_t i = 0; i < td.c.size() / 2; i += 6) // reverse knot order in packets of six
            std::swap_ranges(td.c.begin() + i, td.c.begin() + i + 6, td.c.end() - i - 6); // c++17 only
        return td;
#else
        typename base::data tdsort;
        tdsort.rmax2 = td.rmax2;
        tdsort.rmin2 = td.rmin2;

        // reverse copy all elements in r2
        tdsort.r2.resize(td.r2.size());
        std::reverse_copy(td.r2.begin(), td.r2.end(), tdsort.r2.begin());

        // sanity check before reverse knot copy
        assert(std::is_sorted(td.r2.rbegin(), td.r2.rend()));
        assert(td.c.size() % 6 == 0);
        assert(td.c.size() / (td.r2.size() - 1) == 6);

        // reverse copy knots
        tdsort.c.resize(td.c.size());
        auto dst = tdsort.c.end();
        for (auto src = td.c.begin(); src != td.c.end(); src += 6)
            std::copy(src, src + 6, dst -= 6);
        return tdsort;
#endif
    }
};

} // namespace Tabulate

/**
 * @brief Base class for truncation schemes
 *
 * This is the public interface used for dynamic storage of
 * different schemes. Energy functions are provided as virtual
 * functions which carries runtime overhead. The best performance
 * call these functions directly from the derived class.
 *
 * @fix Should give warning if 'dipolar_selfenergy' is false
 * @warning Charge neutralization scheme is not always implemented (or tested) for Yukawa-type potentials.
 */
class SchemeBase {
  private:
    std::array<double, 2> self_energy_prefactor; // Prefactor for self-energies, UNIT: [ 1 ]

  protected:
    double invcutoff = 0; // inverse cutoff distance, UNIT: [ ( input length )^-1 ]
    double cutoff2 = 0;   // square cutoff distance, UNIT: [ ( input length )^2 ]
    double kappa = 0;     // inverse Debye-length, UNIT: [ ( input length )^-1 ]
    double T0 = 0;        // Spatial Fourier transformed modified interaction tensor, used to calculate the dielectric
                          // constant, UNIT: [ 1 ]
    double chi = 0; // Negative integrated volume potential to neutralize charged system, UNIT: [ ( input length )^2 ]
    bool dipolar_selfenergy = false;             // is there a valid dipolar self-energy?

    void setSelfEnergyPrefactor(const std::array<double, 2> &factor) {
        self_energy_prefactor = factor;
        selfEnergyFunctor = [invcutoff=invcutoff, factor = factor](const std::array<double, 2> &squared_moments) {
          double e_self = 0.0;
          for (int i = 0; i < (int)squared_moments.size(); i++)
              e_self += factor[i] * squared_moments[i] * powi(invcutoff, 2 * i + 1);
          return e_self;
        };
    }

  public:
    std::string doi;     //!< DOI for original citation
    std::string name;    //!< Descriptive name
    Scheme scheme;       //!< Truncation scheme
    double cutoff;       //!< Cut-off distance, UNIT: [ input length ]
    double debye_length; //!< Debye-length, UNIT: [ input length ]

    std::function<double(const std::array<double, 2> &)> selfEnergyFunctor = nullptr; //!< Functor to calc. self-energy

    inline SchemeBase(Scheme scheme, double cutoff, double debye_length = infinity)
        : invcutoff(1.0/cutoff), cutoff2(cutoff*cutoff), kappa(1.0/debye_length), scheme(scheme), cutoff(cutoff), debye_length(debye_length) {}

    virtual ~SchemeBase() = default;

    virtual double neutralization_energy(const std::vector<double> &, double) const { return 0.0; }

    /**
     * @brief Calculate dielectric constant
     * @param M2V see details, UNIT: [ 1 ]
     *
     * @details The dimensionless paramter `M2V` is described by
     * @f$
     *     M2V = \frac{\langle M^2\rangle}{ 3\varepsilon_0Vk_BT }
     * @f$
     * where @f$ \langle M^2\rangle @f$ is mean value of the system dipole moment squared,
     * @f$ \varepsilon_0 @f$ is the vacuum permittivity, _V_ the volume of the system,
     * @f$ k_B @f$ the Boltzmann constant, and _T_ the temperature.
     * When calculating the dielectric constant _T0_ is also needed, i.e. the Spatial Fourier 
     * transformed modified interaction tensor, which is automatically given for each scheme.
     */
    double calc_dielectric(double M2V) { return (M2V * T0 + 2.0 * M2V + 1.0) / (M2V * T0 - M2V + 1.0); }

    virtual double short_range_function(double q) const = 0;
    virtual double short_range_function_derivative(double q) const = 0;
    virtual double short_range_function_second_derivative(double q) const = 0;
    virtual double short_range_function_third_derivative(double q) const = 0;

    virtual double ion_potential(double, double) const = 0;
    virtual double dipole_potential(const vec3 &, const vec3 &) const = 0;
    virtual double quadrupole_potential(const mat33 &, const vec3 &) const = 0;

    virtual double ion_ion_energy(double, double, double) const = 0;
    virtual double ion_dipole_energy(double, const vec3 &, const vec3 &) const = 0;
    virtual double dipole_dipole_energy(const vec3 &, const vec3 &, const vec3 &) const = 0;
    virtual double ion_quadrupole_energy(double, const mat33 &, const vec3 &) const = 0;
    virtual double multipole_multipole_energy(double, double, const vec3 &, const vec3 &, const mat33 &, const mat33 &, const vec3 &) const = 0;

    virtual vec3 ion_field(double, const vec3 &) const = 0;
    virtual vec3 dipole_field(const vec3 &, const vec3 &) const = 0;
    virtual vec3 quadrupole_field(const mat33 &, const vec3 &) const = 0;
    virtual vec3 multipole_field(double, const vec3 &, const mat33 &, const vec3 &) const = 0;

    virtual vec3 ion_ion_force(double, double, const vec3 &) const = 0;
    virtual vec3 ion_dipole_force(double, const vec3 &, const vec3 &) const = 0;
    virtual vec3 dipole_dipole_force(const vec3 &, const vec3 &, const vec3 &) const = 0;
    virtual vec3 ion_quadrupole_force(double, const mat33 &, const vec3 &) const = 0;
    virtual vec3 multipole_multipole_force(double, double, const vec3 &, const vec3 &, const mat33 &, const mat33 &, const vec3 &) const = 0;

    // add remaining funtions here...

    // virtual double surface_energy(const std::vector<vec3> &, const std::vector<double> &, const std::vector<vec3> &,
    // double) const { return 0.0; } virtual vec3 surface_force(const std::vector<vec3> &, const std::vector<double> &,
    // const std::vector<vec3> &, int, double) const { return {0.0,0.0,0.0}; }

#ifdef NLOHMANN_JSON_HPP
  private:
    virtual void _to_json(nlohmann::json &) const = 0;

  public:
    inline void to_json(nlohmann::json &j) const {
        _to_json(j);
        if (std::isfinite(cutoff))
            j["cutoff"] = cutoff;
        if (not doi.empty())
            j["doi"] = doi;
        if (not name.empty())
            j["type"] = name;
        if (std::isfinite(debye_length))
            j["debyelength"] = debye_length;
    }
#endif
};

/**
 * @brief Intermediate base class that implements the interaction energies
 *
 * Derived function need only implement short range functions and derivatives thereof.
 */
template <class T, bool debyehuckel = true> class EnergyImplementation : public SchemeBase {

  public:

    EnergyImplementation(Scheme type, double cutoff, double debyelength = infinity)
        : SchemeBase(type, cutoff, debyelength) {
    }

    /**
     * @brief electrostatic potential from point charge
     * @param z charge, UNIT: [ input charge ]
     * @param r distance from charge, UNIT: [ input length ]
     * @returns ion potential, UNIT: [ ( input charge ) / ( input length ) ]
     *
     * The electrostatic potential from a point charge is described by
     * @f[
     *     \Phi(z,r) = \frac{z}{r}s(q)
     * @f]
     */
    inline double ion_potential(double z, double r) const override {
        if (r < cutoff) {
            double q = r * invcutoff;
            if (debyehuckel) // determined at compile time
                return z / r * static_cast<const T *>(this)->short_range_function(q) * std::exp(-kappa * r);
            else
                return z / r * static_cast<const T *>(this)->short_range_function(q);
        } else {
            return 0.0;
        }
    }

    /**
     * @brief electrostatic potential from point dipole
     * @param mu dipole moment, UNIT: [ ( input length ) x ( input charge ) ]
     * @param r distance vector from dipole, UNIT: [ input length ]
     * @returns dipole potential, UNIT: [ ( input charge ) / ( input length ) ]
     *
     * The potential from a point dipole is described by
     * @f[
     *     \Phi(\boldsymbol{\mu}, {\bf r}) = \frac{\boldsymbol{\mu} \cdot \hat{{\bf r}} }{|{\bf r}|^2} \left( s(q) -
     * qs^{\prime}(q) \right)
     * @f]
     */
    inline double dipole_potential(const vec3 &mu, const vec3 &r) const override {
        double r2 = r.squaredNorm();
        if (r2 < cutoff2) {
            double r1 = std::sqrt(r2);
            double q = r1 * invcutoff;
            double kr = kappa * r1;
            return mu.dot(r) / (r2 * r1) *
                   (static_cast<const T *>(this)->short_range_function(q) * (1.0 + kr) -
                    q * static_cast<const T *>(this)->short_range_function_derivative(q)) *
                   std::exp(-kr);
        } else {
            return 0.0;
        }
    }

    /**
     * @brief electrostatic potential from point dipole
     * @param quad quadrupole moment (not necessarily traceless), UNIT: [ ( input length )^2 x ( input charge ) ]
     * @param r distance vector from dipole, UNIT: [ input length ]
     * @returns quadrupole potential, UNIT: [ ( input charge ) / ( input length ) ]
     *
     * The potential from a point quadrupole is described by
     * @f[
     *     \Phi(\boldsymbol{Q}, {\bf r}) = \frac{1}{2}...
     * @f]
     */
    inline double quadrupole_potential(const mat33 &quad, const vec3 &r) const override {
        double r2 = r.squaredNorm();
        if (r2 < cutoff2) {
            double r1 = std::sqrt(r2);
            double q = r1 * invcutoff;
            double q2 = q * q;
            double kr = kappa * r1;
            double kr2 = kr * kr;
            double srf = static_cast<const T *>(this)->short_range_function(q);
            double dsrf = static_cast<const T *>(this)->short_range_function_derivative(q);
            double ddsrf = static_cast<const T *>(this)->short_range_function_second_derivative(q);

            double a = (srf * (1.0 + kr + kr2 / 3.0) - q * dsrf * (1.0 + 2.0 / 3.0 * kr) + q2 / 3.0 * ddsrf);
            double b = (srf * kr2 - 2.0 * kr * q * dsrf + ddsrf * q2) / 3.0;
            return 0.5 * ( ( 3.0/r2*r.transpose()*quad*r - quad.trace() ) * a + quad.trace() * b ) / r2 / r1 * std::exp(-kappa * r1);
        } else {
            return 0.0;
        }
    }

    /**
     * @brief electrostatic field from point charge
     * @param z point charge, UNIT: [ input charge ]
     * @param r distance-vector from point charge, UNIT: [ input length ]
     * @returns field from charge, UNIT: [ ( input charge ) / ( input length )^2 ]
     *
     * The field from a charge is described by
     * @f[
     *     {\bf E}(z, {\bf r}) = \frac{z \hat{{\bf r}} }{|{\bf r}|^2} \left( s(q) - qs^{\prime}(q) \right)
     * @f].
     */
    inline vec3 ion_field(double z, const vec3 &r) const override {
        double r2 = r.squaredNorm();
        if (r2 < cutoff2) {
            double r1 = std::sqrt(r2);
            double q = r1 * invcutoff;
            double kr = kappa * r1;
            return z * r / (r2 * r1) *
                   (static_cast<const T *>(this)->short_range_function(q) * (1.0 + kr) -
                    q * static_cast<const T *>(this)->short_range_function_derivative(q)) *
                   std::exp(-kr);
        } else {
            return {0, 0, 0};
        }
    }

    /**
     * @brief electrostatic field from point dipole
     * @param mu point dipole, UNIT: [ ( input length ) x ( input charge ) ]
     * @param r distance-vector from point dipole, UNIT: [ input length ]
     * @returns field from dipole, UNIT: [ ( input charge ) / ( input length )^2 ]
     *
     * The field from a point dipole is described by
     * @f[
     *     {\bf E}(\boldsymbol{\mu}, {\bf r}) = \frac{3 ( \boldsymbol{\mu} \cdot \hat{{\bf r}} ) \hat{{\bf r}} -
     * \boldsymbol{\mu} }{|{\bf r}|^3} \left( s(q) - qs^{\prime}(q)  + \frac{q^2}{3}s^{\prime\prime}(q) \right) +
     * \frac{\boldsymbol{\mu}}{|{\bf r}|^3}\frac{q^2}{3}s^{\prime\prime}(q)
     * @f]
     */
    inline vec3 dipole_field(const vec3 &mu, const vec3 &r) const override {
        double r2 = r.squaredNorm();
        if (r2 < cutoff2) {
            double r1 = std::sqrt(r2);
            double r3 = r1 * r2;
            double q = r1 * invcutoff;
            double q2 = q * q;
            double kr = kappa * r1;
            double kr2 = kr * kr;
            double srf = static_cast<const T *>(this)->short_range_function(q);
            double dsrf = static_cast<const T *>(this)->short_range_function_derivative(q);
            double ddsrf = static_cast<const T *>(this)->short_range_function_second_derivative(q);
            vec3 fieldD = (3.0 * mu.dot(r) * r / r2 - mu) / r3;
            fieldD *= (srf * (1.0 + kr + kr2 / 3.0) - q * dsrf * (1.0 + 2.0 / 3.0 * kr) +
                       q2 / 3.0 * ddsrf);
            vec3 fieldI = mu / r3 * (srf * kr2 - 2.0 * kr * q * dsrf + ddsrf * q2) / 3.0;
            return (fieldD + fieldI) * std::exp(-kr);
        } else {
            return {0, 0, 0};
        }
    }

    /**
     * @brief electrostatic field from point quadrupole
     * @param quad point quadrupole, UNIT: [ ( input length )^2 x ( input charge ) ]
     * @param r distance-vector from point quadrupole, UNIT: [ input length ]
     * @returns field from quadrupole, UNIT: [ ( input charge ) / ( input length )^2 ]
     *
     * The field from a point quadrupole is described by
     * @f[
     *     {\bf E}(\boldsymbol{Q}, {\bf r}) = ...
     * @f]
     */
    inline vec3 quadrupole_field(const mat33 &quad, const vec3 &r) const {
        double r2 = r.squaredNorm();
        if (r2 < cutoff2) {
            double r1 = std::sqrt(r2);
            vec3 rh = r / r1;
            double q = r1 * invcutoff;
            double q2 = q * q;
            double kr = kappa * r1;
            double kr2 = kr * kr;
            double r4 = r2 * r2;
            vec3 quadrh = quad*rh;
            vec3 quadTrh = quad.transpose()*rh;
	    double quadfactor = 1.0/r2*r.transpose()*quad*r;
            vec3 fieldD =
                3.0 * ((5.0 * quadfactor - quad.trace()) * rh - quadrh - quadTrh) / r4;
            double srf = static_cast<const T *>(this)->short_range_function(q);
            double dsrf = static_cast<const T *>(this)->short_range_function_derivative(q);
            double ddsrf = static_cast<const T *>(this)->short_range_function_second_derivative(q);
            double dddsrf = static_cast<const T *>(this)->short_range_function_third_derivative(q);
            fieldD *= (srf * (1.0 + kr + kr2 / 3.0) - q * dsrf * (1.0 + 2.0 / 3.0 * kr) + q2 / 3.0 * ddsrf);
            vec3 fieldI = quadfactor * rh / r4;
            fieldI *= (srf * (1.0 + kr) * kr2 - q * dsrf * (3.0 * kr + 2.0) * kr + ddsrf * (1.0 + 3.0 * kr) * q2 - q2 * q * dddsrf);
            return 0.5 * (fieldD + fieldI) * std::exp(-kr);
        } else {
            return {0, 0, 0};
        }
    }

    /**
     * @brief electrostatic field from point multipole
     * @param z charge, UNIT: [ input charge ]
     * @param mu dipole, UNIT: [ ( input length ) x ( input charge ) ]
     * @param quad point quadrupole, UNIT: [ ( input length )^2 x ( input charge ) ]
     * @param r distance-vector from point multipole, UNIT: [ input length ]
     * @returns field from multipole, UNIT: [ ( input charge ) / ( input length )^2 ]
     *
     * The field from a point multipole is described by
     * @f[
     *     {\bf E}(z,\boldsymbol{\mu}, {\bf r}) = \frac{z \hat{{\bf r}} }{|{\bf r}|^2} \left( s(q) - qs^{\prime}(q) \right) +
     * \frac{3 ( \boldsymbol{\mu} \cdot \hat{{\bf r}} ) \hat{{\bf r}} -
     * \boldsymbol{\mu} }{|{\bf r}|^3} \left( s(q) - qs^{\prime}(q)  + \frac{q^2}{3}s^{\prime\prime}(q) \right) +
     * \frac{\boldsymbol{\mu}}{|{\bf r}|^3}\frac{q^2}{3}s^{\prime\prime}(q)
     * @f]
     */
    inline vec3 multipole_field(double z, const vec3 &mu, const mat33 &quad, const vec3 &r) const {
        double r2 = r.squaredNorm();
        if (r2 < cutoff2) {
            double r1 = std::sqrt(r2);
            vec3 rh = r / r1;
            double q = r1 * invcutoff;
            double q2 = q * q;
            double r3 = r1 * r2;
            double kr = kappa * r1;
            double kr2 = kr * kr;
            double quadfactor = 1.0/r2*r.transpose()*quad*r;
            double srf = static_cast<const T *>(this)->short_range_function(q);
            double dsrf = static_cast<const T *>(this)->short_range_function_derivative(q);
            double ddsrf = static_cast<const T *>(this)->short_range_function_second_derivative(q);
            double dddsrf = static_cast<const T *>(this)->short_range_function_third_derivative(q);
            vec3 fieldIon = z * r / r3 * ( srf * (1.0 + kr) - q * dsrf ); // field from ion
             double postfactor = (srf * (1.0 + kr + kr2 / 3.0) - q * dsrf * (1.0 + 2.0 / 3.0 * kr) + q2 / 3.0 * ddsrf);
            vec3 fieldDd = (3.0 * mu.dot(r) * r / r2 - mu) / r3 * postfactor;
            vec3 fieldId = mu / r3 * (srf * kr2 - 2.0 * kr * q * dsrf + ddsrf * q2) / 3.0;
            vec3 fieldDq = 3.0 * ((5.0 * quadfactor - quad.trace()) * rh - quad * rh - quad.transpose() * rh) / r3 / r1 * postfactor;
            vec3 fieldIq = quadfactor * rh / r3 / r1;
            fieldIq *= (srf * (1.0 + kr) * kr2 - q * dsrf * (3.0 * kr + 2.0) * kr + ddsrf * (1.0 + 3.0 * kr) * q2 - q2 * q * dddsrf);
            return ( fieldIon + fieldDd + fieldId + 0.5 * (fieldDq + fieldIq) ) * std::exp(-kr);
        } else {
            return {0, 0, 0};
        }
    }

    /**
     * @brief interaction energy between two point charges
     * @param zA point charge, UNIT: [ input charge ]
     * @param zB point charge, UNIT: [ input charge ]
     * @param r charge-charge separation, UNIT: [ input length ]
     * @returns interaction energy, UNIT: [ ( input charge )^2 / ( input length ) ]
     *
     * The interaction energy between two charges is decribed by
     * @f$
     *     u(z_A, z_B, r) = z_B \Phi(z_A,r)
     * @f$
     * where @f$ \Phi(z_A,r) @f$ is the potential from ion A.
     */
    inline double ion_ion_energy(double zA, double zB, double r) const override { return zB * ion_potential(zA, r); }

    /**
     * @brief interaction energy between a point charges and a point dipole
     * @param z point charge, UNIT: [ input charge ]
     * @param mu dipole moment, UNIT: [ ( input length ) x ( input charge ) ]
     * @param r distance-vector between dipole and charge, @f$ {\bf r} = {\bf r}_{\mu} - {\bf r}_z @f$, UNIT: [ input length ]
     * @returns interaction energy, UNIT: [ ( input charge )^2 / ( input length ) ]
     *
     * The interaction energy between an ion and a dipole is decribed by
     * @f[
     *     u(z, \boldsymbol{\mu}, {\bf r}) = z \Phi(\boldsymbol{\mu}, -{\bf r})
     * @f]
     * where @f$ \Phi(\boldsymbol{\mu}, -{\bf r}) @f$ is the potential from the dipole at the location of the ion.
     * This interaction can also be described by
     * @f[
     *     u(z, \boldsymbol{\mu}, {\bf r}) = -\boldsymbol{\mu}\cdot {\bf E}(z, {\bf r})
     * @f]
     * where @f$ {\bf E}(z, {\bf r}) @f$ is the field from the ion at the location of the dipole.
     */
    inline double ion_dipole_energy(double z, const vec3 &mu, const vec3 &r) const override {
        // Both expressions below gives same answer. Keep for possible optimization in future.
        // return -mu.dot(ion_field(z,r)); // field from charge interacting with dipole
        return z * dipole_potential(mu, -r); // potential of dipole interacting with charge
    }

    /**
     * @brief interaction energy between two point dipoles
     * @param muA dipole moment of particle A, UNIT: [ ( input length ) x ( input charge ) ]
     * @param muB dipole moment of particle B, UNIT: [ ( input length ) x ( input charge ) ]
     * @param r distance-vector between dipoles, @f$ {\bf r} = {\bf r}_{\mu_B} - {\bf r}_{\mu_A} @f$, UNIT: [ input length ]
     * @returns interaction energy, UNIT: [ ( input charge )^2 / ( input length ) ]
     *
     * The interaction energy between two dipoles is decribed by
     * @f[
     *     u(\boldsymbol{\mu}_A, \boldsymbol{\mu}_B, {\bf r}) = -\boldsymbol{\mu}_A\cdot {\bf E}(\boldsymbol{\mu}_B,
     * {\bf r})
     * @f]
     * where @f$ {\bf E}(\boldsymbol{\mu}_B, {\bf r}) @f$ is the field from dipole B at the location of dipole A.
     */
    inline double dipole_dipole_energy(const vec3 &muA, const vec3 &muB, const vec3 &r) const override {
        return -muA.dot(dipole_field(muB, r));
    }

    /**
     * @brief interaction energy between a point charges and a point quadrupole
     * @param z point charge, UNIT: [ input charge ]
     * @param quad quadrupole moment, UNIT: [ ( input length )^2 x ( input charge ) ]
     * @param r distance-vector between quadrupole and charge, @f$ {\bf r} = {\bf r}_{\boldsymbol{Q}} - {\bf r}_z @f$, UNIT: [ input length ]
     * @returns interaction energy, UNIT: [ ( input charge )^2 / ( input length ) ]
     *
     * The interaction energy between an ion and a quadrupole is decribed by
     * @f[
     *     u(z, \boldsymbol{Q}, {\bf r}) = z \Phi(\boldsymbol{Q}, -{\bf r})
     * @f]
     * where @f$ \Phi(\boldsymbol{Q}, -{\bf r}) @f$ is the potential from the quadrupole at the location of the ion.
     */
    inline double ion_quadrupole_energy(double z, const mat33 &quad, const vec3 &r) const override {
        return z * quadrupole_potential(quad, -r); // potential of quadrupole interacting with charge
    }

    /**
     * @brief interaction energy between two multipoles with charges and dipole moments
     * @param zA point charge of particle A, UNIT: [ input charge ]
     * @param zB point charge of particle B, UNIT: [ input charge ]
     * @param muA point dipole moment of particle A, UNIT: [ ( input length ) x ( input charge ) ]
     * @param muB point dipole moment of particle B, UNIT: [ ( input length ) x ( input charge ) ]
     * @param quadA point quadrupole of particle A, UNIT: [ ( input length )^2 x ( input charge ) ]
     * @param quadB point quadrupole of particle B, UNIT: [ ( input length )^2 x ( input charge ) ]
     * @param r distance-vector between dipoles, @f$ {\bf r} = {\bf r}_{\mu_B} - {\bf r}_{\mu_A} @f$, UNIT: [ input length ]
     * @returns interaction energy, UNIT: [ ( input charge )^2 / ( input length ) ]
     *
     * A combination of the functions 'ion_ion_energy', 'ion_dipole_energy', 'dipole_dipole_energy' and 'ion_quadrupole_energy'.
     */
    inline double multipole_multipole_energy(double zA, double zB, const vec3 &muA, const vec3 &muB, const mat33 &quadA, const mat33 &quadB,
                                             const vec3 &r) const override {
        double r2 = r.squaredNorm();
        if (r2 < cutoff2) {
            double r1 = std::sqrt(r2);
            double q = r1 / cutoff;
            double kr = kappa * r1;
            double quadAtrace = quadA.trace();
            double quadBtrace = quadB.trace();

            double srf = static_cast<const T *>(this)->short_range_function(q);
            double dsrfq = static_cast<const T *>(this)->short_range_function_derivative(q) * q;
            double ddsrfq2 = static_cast<const T *>(this)->short_range_function_second_derivative(q) * q * q / 3.0;

            double angcor = (srf * (1.0 + kr) - dsrfq);
            double unicor = (srf * kr * kr / 3.0 - 2.0 / 3.0 * dsrfq * kr + ddsrfq2);
	    double muBdotr = muB.dot(r);
            vec3 field_dipoleB = (3.0 * muBdotr * r / r2 - muB) * (angcor + unicor) + muB * unicor;

            double ion_ion = zA * zB * srf * r2; // will later be divided by r3
            double ion_dipole = (zB * muA.dot(r) - zA * muBdotr) * angcor; // will later be divided by r3
            double dipole_dipole = -muA.dot(field_dipoleB); // will later be divided by r3
            double ion_quadrupole = zA * 0.5 * ( ( 3.0/r2*r.transpose()*quadB*r - quadBtrace ) * (angcor + unicor) + quadBtrace * unicor ); // will later be divided by r3
            ion_quadrupole += zB * 0.5 * ( ( 3.0/r2*r.transpose()*quadA*r - quadAtrace ) * (angcor + unicor) + quadAtrace * unicor );

            return (ion_ion + ion_dipole + dipole_dipole + ion_quadrupole) * std::exp(-kr) / r2 / r1;
        } else {
            return 0.0;
        }
    }

    /**
     * @brief interaction force between two point charges
     * @param zA point charge, UNIT: [ input charge ]
     * @param zB point charge, UNIT: [ input charge ]
     * @param r distance-vector between charges, @f$ {\bf r} = {\bf r}_{z_B} - {\bf r}_{z_A} @f$, UNIT: [ input length ]
     * @returns interaction force, UNIT: [ ( input charge )^2 / ( input length )^2 ]
     *
     * The force between two point charges is described by
     * @f[
     *     {\bf F}(z_A, z_B, {\bf r}) = z_B {\bf E}(z_A, {\bf r})
     * @f]
     * where @f$ {\bf E}(z_A, {\bf r}) @f$ is the field from ion A at the location of ion B.
     */
    inline vec3 ion_ion_force(double zA, double zB, const vec3 &r) const override { return zB * ion_field(zA, r); }

    /**
     * @brief interaction force between a point charges and a point dipole
     * @param z charge, UNIT: [ input charge ]
     * @param mu dipole moment, UNIT: [ ( input length ) x ( input charge ) ]
     * @param r distance-vector between dipole and charge, @f$ {\bf r} = {\bf r}_{\mu} - {\bf r}_z @f$, UNIT: [ input length ]
     * @returns interaction force, UNIT: [ ( input charge )^2 / ( input length )^2 ]
     *
     * @details The force between an ion and a dipole is decribed by
     * @f[
     *     {\bf F}(z, \boldsymbol{\mu}, {\bf r}) = z {\bf E}(\boldsymbol{\mu}, {\bf r})
     * @f]
     * where @f$ {\bf E}(\boldsymbol{\mu}, {\bf r}) @f$ is the field from the dipole at the location of the ion.
     */
    inline vec3 ion_dipole_force(double z, const vec3 &mu, const vec3 &r) const override {
        return z * dipole_field(mu, r);
    }

    /**
     * @brief interaction force between two point dipoles
     * @param muA dipole moment of particle A, UNIT: [ ( input length ) x ( input charge ) ]
     * @param muB dipole moment of particle B, UNIT: [ ( input length ) x ( input charge ) ]
     * @param r distance-vector between dipoles, @f[ {\bf r} = {\bf r}_{\mu_B} - {\bf r}_{\mu_A} @f], UNIT: [ input length ]
     * @returns interaction force, UNIT: [ ( input charge )^2 / ( input length )^2 ]
     *
     * @details The force between two dipoles is decribed by
     * @f[
     *     {\bf F}(\boldsymbol{\mu}_A, \boldsymbol{\mu}_B, {\bf r}) = {\bf F}_D(\boldsymbol{\mu}_A, \boldsymbol{\mu}_B,
     * {\bf r})\left( s(q) - qs^{\prime}(q)  + \frac{q^2}{3}s^{\prime\prime}(q) \right) + {\bf F}_I(\boldsymbol{\mu}_A,
     * \boldsymbol{\mu}_B, {\bf r})\left( s^{\prime\prime}(q)  - qs^{\prime\prime\prime}(q) \right)q^2
     * @f]
     * where the 'direct' (D) force contribution is
     * @f[
     *     {\bf F}_D(\boldsymbol{\mu}_A, \boldsymbol{\mu}_B, {\bf r}) = 3\frac{ 5 (\boldsymbol{\mu}_A \cdot {\bf
     * \hat{r}}) (\boldsymbol{\mu}_B \cdot {\bf \hat{r}}){\bf \hat{r}} - (\boldsymbol{\mu}_A \cdot
     * \boldsymbol{\mu}_B){\bf \hat{r}} - (\boldsymbol{\mu}_A \cdot {\bf \hat{r}})\boldsymbol{\mu}_B -
     * (\boldsymbol{\mu}_B \cdot {\bf \hat{r}})\boldsymbol{\mu}_A }{|{\bf r}|^4}
     * @f]
     * and the 'indirect' (I) force contribution is
     * @f[
     *     {\bf F}_I(\boldsymbol{\mu}_A, \boldsymbol{\mu}_B, {\bf r}) = \frac{ (\boldsymbol{\mu}_A \cdot {\bf \hat{r}})
     * (\boldsymbol{\mu}_B \cdot {\bf \hat{r}}){\bf \hat{r}}}{|{\bf r}|^4}.
     * @f]
     */
    inline vec3 dipole_dipole_force(const vec3 &muA, const vec3 &muB, const vec3 &r) const override {
        double r2 = r.squaredNorm();
        if (r2 < cutoff2) {
            double r1 = std::sqrt(r2);
            vec3 rh = r / r1;
            double q = r1 * invcutoff;
            double q2 = q * q;
            double kr = kappa * r1;
            double r4 = r2 * r2;
            double muAdotRh = muA.dot(rh);
            double muBdotRh = muB.dot(rh);
            vec3 forceD =
                3.0 * ((5.0 * muAdotRh * muBdotRh - muA.dot(muB)) * rh - muBdotRh * muA - muAdotRh * muB) / r4;
            double srf = static_cast<const T *>(this)->short_range_function(q);
            double dsrf = static_cast<const T *>(this)->short_range_function_derivative(q);
            double ddsrf = static_cast<const T *>(this)->short_range_function_second_derivative(q);
            double dddsrf = static_cast<const T *>(this)->short_range_function_third_derivative(q);
            forceD *= (srf * (1.0 + kr + kr * kr / 3.0) - q * dsrf * (1.0 + 2.0 / 3.0 * kr) + q2 / 3.0 * ddsrf);
            vec3 forceI = muAdotRh * muBdotRh * rh / r4;
            forceI *= (srf * (1.0 + kr) * kr * kr - q * dsrf * (3.0 * kr + 2.0) * kr + ddsrf * (1.0 + 3.0 * kr) * q2 - q2 * q * dddsrf);
            return (forceD + forceI) * std::exp(-kr);
        } else {
            return {0, 0, 0};
        }
    }

    /**
     * @brief interaction force between a point charge and a point quadrupole
     * @param z point charge, UNIT: [ input charge ]
     * @param quad point quadrupole, UNIT: [ ( input length )^2 x ( input charge ) ]
     * @param r distance-vector between particles, @f$ {\bf r} = {\bf r}_{Q} - {\bf r}_{z} @f$, UNIT: [ input length ]
     * @returns interaction force, UNIT: [ ( input charge )^2 / ( input length )^2 ]
     *
     * The force between a point charge and a point quadrupole is described by
     * @f[
     *     {\bf F}(z, Q, {\bf r}) = z {\bf E}(Q, {\bf r})
     * @f]
     * where @f$ {\bf E}(Q, {\bf r}) @f$ is the field from the quadrupole at the location of the ion.
     */
    inline vec3 ion_quadrupole_force(double z, const mat33 &quad, const vec3 &r) const override { return z * quadrupole_field(quad, r); }

    /**
     * @brief interaction force between two point multipoles
     * @param zA charge of particle A, UNIT: [ input charge ]
     * @param zB charge of particle B, UNIT: [ input charge ]
     * @param muA dipole moment of particle A, UNIT: [ ( input length ) x ( input charge ) ]
     * @param muB dipole moment of particle B, UNIT: [ ( input length ) x ( input charge ) ]
     * @param quadA point quadrupole of particle A, UNIT: [ ( input length )^2 x ( input charge ) ]
     * @param quadB point quadrupole of particle B, UNIT: [ ( input length )^2 x ( input charge ) ]
     * @param r distance-vector between dipoles, @f[ {\bf r} = {\bf r}_{\mu_B} - {\bf r}_{\mu_A} @f], UNIT: [ input length ]
     * @returns interaction force, UNIT: [ ( input charge )^2 / ( input length )^2 ]
     *
     * @details A combination of the functions 'ion_ion_force', 'ion_dipole_force', 'dipole_dipole_force' and 'ion_quadrupole_force'.
     */
    inline vec3 multipole_multipole_force(double zA, double zB, const vec3 &muA, const vec3 &muB, const mat33 &quadA, const mat33 &quadB,
                                          const vec3 &r) const override {
        double r2 = r.squaredNorm();
        if (r2 < cutoff2) {
            double r1 = std::sqrt(r2);
            double q = r1 * invcutoff;
            double q2 = q * q;
            double kr = kappa * r1;
            vec3 rh = r / r1;
            double muAdotRh = muA.dot(rh);
            double muBdotRh = muB.dot(rh);

            double srf = static_cast<const T *>(this)->short_range_function(q);
            double dsrfq = static_cast<const T *>(this)->short_range_function_derivative(q) * q;
            double ddsrfq2 = static_cast<const T *>(this)->short_range_function_second_derivative(q) * q2 / 3.0;
            double dddsrfq3 = static_cast<const T *>(this)->short_range_function_third_derivative(q) * q2 * q;

            double angcor = (srf * (1.0 + kr) - dsrfq);
            double unicor = (srf * kr - 2.0 * dsrfq) * kr / 3.0 + ddsrfq2;
            double totcor = unicor + angcor;
	    double r3corr = ( angcor * kr * kr - dsrfq * 2.0 * (1.0 + kr) * kr + 3.0 * ddsrfq2 * (1.0 + 3.0 * kr) - dddsrfq3);

            vec3 ion_ion = zB * zA * r * angcor * r1;
            vec3 ion_dipole = zA * ((3.0 * muBdotRh * rh - muB) * totcor + muB * unicor);
            ion_dipole += zB * ((3.0 * muAdotRh * rh - muA) * totcor + muA * unicor);
            ion_dipole *= r1;
            vec3 forceD = 3.0 * ((5.0 * muAdotRh * muBdotRh - muA.dot(muB)) * rh - muBdotRh * muA - muAdotRh * muB) * totcor;
            vec3 dipole_dipole = (forceD + muAdotRh * muBdotRh * rh * r3corr);
            double quadfactor = 1.0/r2*r.transpose()*quadB*r;
            vec3 fieldD = 3.0 * (-(5.0 * quadfactor - quadB.trace()) * rh + quadB*rh + quadB.transpose()*rh) * totcor;
            vec3 ion_quadrupole = zA * 0.5 * (fieldD - quadfactor * rh * r3corr);
            quadfactor = 1.0/r2*r.transpose()*quadA*r;
            fieldD = 3.0 * ((5.0 * quadfactor - quadA.trace()) * rh - quadA*rh - quadA.transpose()*rh) * totcor;
            ion_quadrupole += zB * 0.5 * (fieldD + quadfactor * rh * r3corr);

            return (ion_ion + ion_dipole + dipole_dipole + ion_quadrupole) * std::exp(-kr) / r2 / r2;
        } else {
            return {0, 0, 0};
        }
    }

    /**
     * @brief torque exerted on point dipole due to field
     * @param mu dipole moment, UNIT: [ ( input length ) x ( input charge ) ]
     * @param E field, UNIT: [ ( input charge unit ) / ( input length unit )^2 ]
     * @returns torque, UNIT: [ ( input charge )^2 / ( input length ) ]
     *
     * @details The torque on a dipole in a field is described by
     * @f$
     *     \boldsymbol{\tau} = \boldsymbol{\mu} \times \boldsymbol{E}
     * @f$
     * 
     * @warning Not tested!
     */
    inline vec3 dipole_torque(const vec3 &mu, const vec3 &E) const { return mu.cross(E); }

    /**
     * @brief Self-energy for all type of interactions
     * @param squared_moments vector with square moments, i.e. charge squared and dipole moment squared, UNIT: [ ( input charge )^2 , ( input length )^2 x ( input charge )^2 , ... ]
     * @returns self-energy, UNIT: [ ( input charge )^2 / ( input length ) ]
     *
     * @details The self-energy is described by
     * @f$
     *     u_{self} = p_1 \frac{z^2}{R_c} + p_2 \frac{|\boldsymbol{\mu}|^2}{R_c^3} + \cdots
     * @f$
     * where @f$ p_i @f$ is the prefactor for the self-energy for species 'i'.
     * Here i=0 represent ions, i=1 represent dipoles etc.
     */
    inline double self_energy(const std::array<double, 2> &squared_moments) const {
        assert(selfEnergyFunctor != nullptr);
        return selfEnergyFunctor(squared_moments);
    }

    /**
     * @brief Compensating term for non-neutral systems
     * @param charges Charges of particles, UNIT: [ input charge ]
     * @param volume Volume of unit-cell, UNIT: [ ( input length )^3 ]
     * @returns energy, UNIT: [ ( input charge )^2 / ( input length ) ]
     * @note DOI:10.1021/jp951011v
     * @warning Not tested!
     */
    inline double neutralization_energy(const std::vector<double> &charges, double volume) const override {
        double charge_total = 0.0;
        for (unsigned int i = 0; i < charges.size(); i++)
            charge_total += charges.at(i);
        return ((this)->chi / 2.0 / volume * charge_total * charge_total);
    }
};

// -------------- Plain ---------------

/**
 * @brief No truncation scheme, cutoff = infinity
 * @warning Neutralization-scheme should not be used using an infinite cut-off
 */
class Plain : public EnergyImplementation<Plain> {
  public:
    inline Plain(double debye_length = infinity)
        : EnergyImplementation(Scheme::plain, std::sqrt(std::numeric_limits<double>::max()), debye_length) {
        name = "plain";
        dipolar_selfenergy = true;
        doi = "Premier mémoire sur l’électricité et le magnétisme by Charles-Augustin de Coulomb"; // :P
        setSelfEnergyPrefactor({0.0, 0.0});
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
        chi = -2.0 * std::acos(-1.0) * cutoff2; // should not be used!
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

// -------------- Ewald real-space (using Gaussian) ---------------

/**
 * @brief Ewald real-space scheme using a Gaussian screening-function.
 *
 * @note The implemented charge-compensation for Ewald differes from that of in DOI:10.1021/ct400626b where chi = -pi /
 * alpha2. This expression is only correct if integration is over all space, not just the cutoff region, cf. Eq. 14
 * in 10.1021/jp951011v. Thus the implemented expression is roughly -pi / alpha2 for alpha > ~2-3. User beware!
 * (also see DOI:10.1063/1.470721)
 */
class Ewald : public EnergyImplementation<Ewald> {
    double eta, eta2, eta3;                //!< Reduced damping-parameter, and squared, and cubed
    double zeta, zeta2, zeta3;             //!< Reduced inverse Debye-length, and squared, and cubed
    double eps_sur;                        //!< Dielectric constant of the surrounding medium
    const double pi_sqrt = 2.0 * std::sqrt(std::atan(1.0));
    const double pi = 4.0 * std::atan(1.0);

  public:
    /**
     * @param cutoff distance cutoff
     * @param alpha damping-parameter
     */
    inline Ewald(double cutoff, double alpha, double eps_sur = infinity, double debye_length = infinity)
        : EnergyImplementation(Scheme::ewald, cutoff), eps_sur(eps_sur) {
        name = "Ewald real-space";
        dipolar_selfenergy = true;
        doi = "10.1002/andp.19213690304";
        eta = alpha * cutoff;
        eta2 = eta * eta;
        eta3 = eta2 * eta;
        if (eps_sur < 1.0)
            eps_sur = infinity;
        double Q = 1.0 - std::erfc(eta) - 2.0 * eta / pi_sqrt * std::exp(-eta2); // Eq. 12 in DOI: 10.1016/0009-2614(83)80585-5 using 'K = cutoff region'
        T0 = (std::isinf(eps_sur)) ? Q : ( Q - 1.0 + 2.0 * (eps_sur - 1.0) / (2.0 * eps_sur + 1.0) ); // Eq. 17 in DOI: 10.1016/0009-2614(83)80585-5
        zeta = cutoff / debye_length;
        zeta2 = zeta * zeta;
        zeta3 = zeta2 * zeta;
        if(zeta < 1e-6) {
            chi = -pi * cutoff2 * ( 1.0 - std::erfc( eta ) * ( 1.0 - 2.0 * eta2 ) - 2.0 * eta * std::exp( -eta2 ) / pi_sqrt ) / eta2;
        } else {
            chi = 4.0 * ( 0.5 * ( 1.0 - zeta ) * std::erfc( eta + zeta / ( 2.0 * eta ) ) * std::exp( zeta ) + std::erf( eta ) * std::exp(-zeta2 / ( 4.0 * eta2 ) ) + 
                0.5 * ( 1.0 + zeta ) * std::erfc( eta - zeta / ( 2.0 * eta ) ) * std::exp( -zeta ) - 1.0 ) * pi * cutoff2 / zeta2;
        }
        // chi = -pi * cutoff2 / eta2 according to DOI:10.1021/ct400626b, for uncscreened system

        setSelfEnergyPrefactor({
            -eta / pi_sqrt * (std::exp(-zeta2 / 4.0 / eta2) - pi_sqrt * zeta / (2.0 * eta) * std::erfc(zeta / (2.0 * eta) ) ),
            -eta3 / pi_sqrt * 2.0 / 3.0 *
                (pi_sqrt * zeta3 / 4.0 / eta3 * std::erfc(zeta / (2.0 * eta) ) + (1.0 - zeta2 / 2.0 / eta2) * std::exp(-zeta2 / 4.0 / eta2))}); // ion-quadrupole self-energy term: XYZ
    }

    inline double short_range_function(double q) const override {
        return 0.5 * (std::erfc(eta * q + zeta / (2.0 * eta)) * std::exp(2.0 * zeta * q) + std::erfc(eta * q - zeta / (2.0 * eta)));
    }
    inline double short_range_function_derivative(double q) const override {
        double expC = std::exp(-powi(eta * q - zeta / (2.0 * eta), 2));
        double erfcC = std::erfc(eta * q + zeta / (2.0 * eta));
        return (-2.0 * eta / pi_sqrt * expC + zeta * erfcC * std::exp(2.0 * zeta * q));
    }
    inline double short_range_function_second_derivative(double q) const override {
        double expC = std::exp(-powi(eta * q - zeta / (2.0 * eta), 2));
        double erfcC = std::erfc(eta * q + zeta / (2.0 * eta));
        return (4.0 * eta2 / pi_sqrt * (eta * q - zeta / eta) * expC + 2.0 * zeta2 * erfcC * std::exp(2.0 * zeta * q));
    }
    inline double short_range_function_third_derivative(double q) const override {
        double expC = std::exp(-powi(eta * q - zeta / (2.0 * eta), 2));
        double erfcC = std::erfc(eta * q + zeta / (2.0 * eta));
        return (4.0 * eta3 / pi_sqrt * (1.0 - 2.0 * (eta * q - zeta / eta) * (eta * q - zeta / (2.0 * eta) ) - zeta2 / eta2) * expC + 4.0 * zeta3 * erfcC * std::exp(2.0 * zeta * q));
    }

    /**
     * @brief Reciprocal-space energy
     * @param positions Positions of particles
     * @param charges Charges of particles
     * @param dipoles Dipole moments of particles
     * @param L Dimensions of unit-cell
     * @param nmax Cut-off in reciprocal-space
     * @note Uses spherical cut-off in summation
     */
    inline double reciprocal_energy(const std::vector<vec3> &positions, const std::vector<double> &charges,
                                    const std::vector<vec3> &dipoles, const vec3 &L, int nmax) const {

        assert(positions.size() == charges.size());
        assert(positions.size() == dipoles.size());

        double volume = L[0] * L[1] * L[2];
        std::vector<vec3> kvec;
        std::vector<double> Ak;
        // kvec.reserve( expected_size_of_kvec ); // speeds up push_back below
        // Ak.reserve( expected_size_of_Ak ); // speeds up push_back below
        for (int nx = -nmax; nx < nmax + 1; nx++) {
            for (int ny = -nmax; ny < nmax + 1; ny++) {
                for (int nz = -nmax; nz < nmax + 1; nz++) {
                    vec3 kv = {2.0 * pi * nx / L[0], 2.0 * pi * ny / L[1], 2.0 * pi * nz / L[2]};
                    double k2 = kv.squaredNorm() + zeta2 / cutoff2;
                    vec3 nv = {double(nx), double(ny), double(nz)};
                    double nv1 = nv.squaredNorm();
                    if (nv1 > 0) {
                        if (nv1 <= nmax * nmax) {
                            kvec.push_back(kv);
                            Ak.push_back(std::exp(-( k2 * cutoff2 + zeta2 ) / 4.0 / eta2 ) / k2);
                        }
                    }
                }
            }
        }

        assert(kvec.size() == Ak.size());

        double E = 0.0;
        for (size_t k = 0; k < kvec.size(); k++) {
            std::complex<double> Qq(0.0, 0.0);
            std::complex<double> Qmu(0.0, 0.0);
            for (size_t i = 0; i < positions.size(); i++) {
                double kDotR = kvec[k].dot(positions[i]);
                double coskDotR = std::cos(kDotR);
                double sinkDotR = std::sin(kDotR);
                Qq += charges[i] * std::complex<double>(coskDotR, sinkDotR);
                Qmu += dipoles[i].dot(kvec[k]) * std::complex<double>(-sinkDotR, coskDotR);
            }
            std::complex<double> Q = Qq + Qmu;
            E += (powi(std::abs(Q), 2) * Ak[k]);
        }
        return (E * 2.0 * pi / volume);
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
        vec3 sum_r_charges = {0.0, 0.0, 0.0};
        vec3 sum_dipoles = {0.0, 0.0, 0.0};
        for (size_t i = 0; i < positions.size(); i++) {
            sum_r_charges += positions[i] * charges[i];
            sum_dipoles += dipoles[i];
        }
        double sqDipoles =
            sum_r_charges.dot(sum_r_charges) + 2.0 * sum_r_charges.dot(sum_dipoles) + sum_dipoles.dot(sum_dipoles);

        return (2.0 * pi / (2.0 * eps_sur + 1.0) / volume * sqDipoles);
    }

#ifdef NLOHMANN_JSON_HPP
    inline Ewald(const nlohmann::json &j)
        : Ewald(j.at("cutoff").get<double>(), j.at("alpha").get<double>(), j.value("epss", infinity),
                j.value("debyelength", infinity)) {}

  private:
    inline void _to_json(nlohmann::json &j) const override {
        j = {{ "alpha", eta/cutoff }};
        if (std::isinf(eps_sur))
            j["epss"] = "inf";
        else
            j["epss"] = eps_sur;
    }
#endif
};

// -------------- Ewald real-space (using truncated Gaussian) ---------------

/**
 * @brief Ewald real-space scheme using a truncated Gaussian screening-function.
 */
class EwaldT : public EnergyImplementation<EwaldT> {
    double eta, eta2, eta3;                //!< Reduced damping-parameter, and squared, and cubed
    double zeta, zeta2, zeta3;             //!< Reduced inverse Debye-length, and squared, and cubed
    double eps_sur;                        //!< Dielectric constant of the surrounding medium
    double F0;                             //!< 'scaling' of short-ranged function
    const double pi_sqrt = 2.0 * std::sqrt(std::atan(1.0));
    const double pi = 4.0 * std::atan(1.0);

  public:
    /**
     * @param cutoff distance cutoff
     * @param alpha damping-parameter
     */
    inline EwaldT(double cutoff, double alpha, double eps_sur = infinity, double debye_length = infinity)
        : EnergyImplementation(Scheme::ewaldt, cutoff), eps_sur(eps_sur) {
        name = "EwaldT real-space";
        dipolar_selfenergy = true;
        doi = "XYZ";
        eta = alpha * cutoff;
        eta2 = eta * eta;
        eta3 = eta2 * eta;
        if (eps_sur < 1.0)
            eps_sur = infinity;
	F0 = 1.0 - std::erfc(eta) - 2.0 * eta / pi_sqrt * std::exp(-eta2);
        T0 = (std::isinf(eps_sur)) ? 1.0 : 2.0 * (eps_sur - 1.0) / (2.0 * eps_sur + 1.0);
	chi = -( 1.0 - 4.0 * eta3 * std::exp( -eta2 ) / ( 3.0 * pi_sqrt * F0 ) ) * cutoff2 * pi / eta2;
        setSelfEnergyPrefactor({-eta / pi_sqrt * (1.0 - std::exp( -eta2 ) ) / F0, -eta3 * 2.0 / 3.0 / ( std::erf( eta ) * pi_sqrt - 2.0 * eta * std::exp( -eta2 ) ) });
    }

    inline double short_range_function(double q) const override {
	return ( std::erfc(eta * q) - std::erfc(eta) - (1.0 - q) * 2.0 * eta / pi_sqrt * std::exp(-eta2) ) / F0;
    }
    inline double short_range_function_derivative(double q) const override {
	return - 2.0 * eta * ( std::exp( -eta2 * q * q ) - std::exp( -eta2 ) ) / pi_sqrt / F0;
    }
    inline double short_range_function_second_derivative(double q) const override {
        return 4.0 * eta3 * q * std::exp( -eta2 * q * q ) / pi_sqrt / F0;
    }
    inline double short_range_function_third_derivative(double q) const override {
        return - 8.0 * ( eta2 * q * q - 0.5 ) * eta3 * std::exp( -eta2 * q * q ) / pi_sqrt / F0;
    }

    /**
     * @brief Reciprocal-space energy
     * @param positions Positions of particles
     * @param charges Charges of particles
     * @param dipoles Dipole moments of particles
     * @param L Dimensions of unit-cell
     * @param nmax Cut-off in reciprocal-space
     * @note Uses spherical cut-off in summation
     */
    inline double reciprocal_energy(const std::vector<vec3> &positions, const std::vector<double> &charges,
                                    const std::vector<vec3> &dipoles, const vec3 &L, int nmax) const {

        assert(positions.size() == charges.size());
        assert(positions.size() == dipoles.size());

        double volume = L[0] * L[1] * L[2];
        std::vector<vec3> kvec;
        std::vector<double> Ak;
        // kvec.reserve( expected_size_of_kvec ); // speeds up push_back below
        // Ak.reserve( expected_size_of_Ak ); // speeds up push_back below
        for (int nx = -nmax; nx < nmax + 1; nx++) {
            for (int ny = -nmax; ny < nmax + 1; ny++) {
                for (int nz = -nmax; nz < nmax + 1; nz++) {
                    vec3 kv = {2.0 * pi * nx / L[0], 2.0 * pi * ny / L[1], 2.0 * pi * nz / L[2]};
                    double k2 = kv.squaredNorm();// + kappa2;
                    vec3 nv = {double(nx), double(ny), double(nz)};
                    double nv1 = nv.squaredNorm();
                    if (nv1 > 0) {
                        if (nv1 <= nmax * nmax) {
                            kvec.push_back(kv);
			    double kR = std::sqrt(k2) * cutoff;
			    std::complex<double> expV( std::cos( kR ) , -std::sin( kR ) );
			    std::complex<double> z( -kR / ( 2.0 * eta ) , eta );
			    double omegaSin = ( Faddeeva::w(z) * expV ).real();
			    omegaSin += std::sin( kR ) / kR * 2.0 * eta / pi_sqrt;
			    double expVar = std::exp( -k2 * cutoff2 / 4.0 / eta2 ) - omegaSin * std::exp(-eta2);
			    Ak.push_back( expVar / F0 / k2 );
                        }
                    }
                }
            }
        }

        assert(kvec.size() == Ak.size());

        double E = 0.0;
        for (size_t k = 0; k < kvec.size(); k++) {
            std::complex<double> Qq(0.0, 0.0);
            std::complex<double> Qmu(0.0, 0.0);
            for (size_t i = 0; i < positions.size(); i++) {
                double kDotR = kvec[k].dot(positions[i]);
                double coskDotR = std::cos(kDotR);
                double sinkDotR = std::sin(kDotR);
                Qq += charges[i] * std::complex<double>(coskDotR, sinkDotR);
                Qmu += dipoles[i].dot(kvec[k]) * std::complex<double>(-sinkDotR, coskDotR);
            }
            std::complex<double> Q = Qq + Qmu;
            E += (powi(std::abs(Q), 2) * Ak[k]);
        }
        return (E * 2.0 * pi / volume);
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
        vec3 sum_r_charges = {0.0, 0.0, 0.0};
        vec3 sum_dipoles = {0.0, 0.0, 0.0};
        for (size_t i = 0; i < positions.size(); i++) {
            sum_r_charges += positions[i] * charges[i];
            sum_dipoles += dipoles[i];
        }
        double sqDipoles =
            sum_r_charges.dot(sum_r_charges) + 2.0 * sum_r_charges.dot(sum_dipoles) + sum_dipoles.dot(sum_dipoles);

        return (2.0 * pi / (2.0 * eps_sur + 1.0) / volume * sqDipoles);
    }

#ifdef NLOHMANN_JSON_HPP
    inline EwaldT(const nlohmann::json &j)
        : EwaldT(j.at("cutoff").get<double>(), j.at("alpha").get<double>(), j.value("epss", infinity),
                j.value("debyelength", infinity)) {}

  private:
    inline void _to_json(nlohmann::json &j) const override {
        j = {{ "alpha", eta/cutoff }};
        if (std::isinf(eps_sur))
            j["epss"] = "inf";
        else
            j["epss"] = eps_sur;
    }
#endif
};

// -------------- Reaction-field ---------------

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
        dipolar_selfenergy = true;
        doi = "10.1080/00268977300102101";
        //epsRF = epsRF;
        //epsr = epsr;
        //shifted = shifted;
        setSelfEnergyPrefactor({-3.0 * epsRF * double(shifted) / (4.0 * epsRF + 2.0 * epsr),
                                 -(2.0 * epsRF - 2.0 * epsr) / (2.0 * (2.0 * epsRF + epsr))});
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
        j = {{"epsr", epsr}, {"epsRF", epsRF}, { "shifted", shifted }};
    }
#endif
};

// -------------- Zahn ---------------

/**
 * @brief Zahn scheme
 */
class Zahn : public EnergyImplementation<Zahn> {
  private:
    double alpha;               //!< Damping-parameter
    double alphaRed, alphaRed2; //!< Reduced damping-parameter, and squared
    const double pi_sqrt = 2.0 * std::sqrt(std::atan(1.0));
    const double pi = 4.0 * std::atan(1.0);

  public:
    /**
     * @brief Contructor
     * @param cutoff distance cutoff
     * @param alpha damping-parameter
     */
    inline Zahn(double cutoff, double alpha) : EnergyImplementation(Scheme::zahn, cutoff), alpha(alpha) {
        name = "Zahn";
        dipolar_selfenergy = false;
        doi = "10.1021/jp025949h";
        alphaRed = alpha * cutoff;
        alphaRed2 = alphaRed * alphaRed;
        setSelfEnergyPrefactor({-alphaRed * (1.0 - std::exp(-alphaRed2)) / pi_sqrt + 0.5 * std::erfc(alphaRed),
                                 0.0}); // Dipole self-energy undefined!
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
    inline void _to_json(nlohmann::json &j) const override {
        j = {{ "alpha", alpha }};
    }
#endif
};

// -------------- Fennell ---------------

/**
 * @brief Fennell scheme
 */
class Fennell : public EnergyImplementation<Fennell> {
  private:
    double alpha;               //!< Damping-parameter
    double alphaRed, alphaRed2; //!< Reduced damping-parameter, and squared
    const double pi_sqrt = 2.0 * std::sqrt(std::atan(1.0));
    const double pi = 4.0 * std::atan(1.0);

  public:
    /**
     * @param cutoff distance cutoff
     * @param alpha damping-parameter
     */
    inline Fennell(double cutoff, double alpha) : EnergyImplementation(Scheme::fennell, cutoff), alpha(alpha) {
        name = "Fennell";
        dipolar_selfenergy = false;
        doi = "10.1063/1.2206581";
        alphaRed = alpha * cutoff;
        alphaRed2 = alphaRed * alphaRed;
        setSelfEnergyPrefactor({-alphaRed * (1.0 + std::exp(-alphaRed2)) / pi_sqrt - std::erfc(alphaRed),
                                 0.0}); // Dipole self-energy undefined!
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
        chi = (2.0 * ((alphaRed2 + 3.0) * alphaRed * std::exp(-alphaRed2) +
                      0.5 * (std::erf(alphaRed) * alphaRed2 - alphaRed2 - 3.0 * std::erf(alphaRed)) * std::sqrt(pi))) *
              std::sqrt(pi) * cutoff * cutoff / (3.0 * alphaRed2);
    }

    inline double short_range_function(double q) const override {
        return (std::erfc(alphaRed * q) - q * std::erfc(alphaRed) +
                (q - 1.0) * q * (std::erfc(alphaRed) + 2.0 * alphaRed * std::exp(-alphaRed2) / pi_sqrt));
    }
    inline double short_range_function_derivative(double q) const override {
        return (2.0 * alphaRed * (2.0 * (q - 0.5) * std::exp(-alphaRed2) - std::exp(-alphaRed2 * q * q)) / pi_sqrt +
                2.0 * std::erfc(alphaRed) * (q - 1.0));
    }
    inline double short_range_function_second_derivative(double q) const override {
        return (4.0 * alphaRed * (alphaRed2 * q * std::exp(-alphaRed2 * q * q) + std::exp(-alphaRed2)) / pi_sqrt +
                2.0 * std::erfc(alphaRed));
    }
    inline double short_range_function_third_derivative(double q) const override {
        return 4.0 * alphaRed2 * alphaRed * (1.0 - 2.0 * alphaRed2 * q * q) * std::exp(-alphaRed2 * q * q) / pi_sqrt;
    }

#ifdef NLOHMANN_JSON_HPP
    /** Construct from JSON object, looking for `cutoff`, `alpha` */
    inline Fennell(const nlohmann::json &j) : Fennell(j.at("cutoff").get<double>(), j.at("alpha").get<double>()) {}

  private:
    inline void _to_json(nlohmann::json &j) const override {
        j = {{ "alpha", alpha }};
    }
#endif
};

// -------------- Zero-dipole ---------------

/**
 * @brief Zero-dipole scheme
 */
class ZeroDipole : public EnergyImplementation<ZeroDipole> {
  private:
    double alpha;               //!< Damping-parameter
    double alphaRed, alphaRed2; //!< Reduced damping-parameter, and squared
    const double pi_sqrt = 2.0 * std::sqrt(std::atan(1.0));
    const double pi = 4.0 * std::atan(1.0);

  public:
    /**
     * @param cutoff distance cutoff
     * @param alpha damping-parameter
     */
    inline ZeroDipole(double cutoff, double alpha) : EnergyImplementation(Scheme::zerodipole, cutoff), alpha(alpha) {
        name = "ZeroDipole";
        dipolar_selfenergy = true;
        doi = "10.1063/1.3582791";
        alphaRed = alpha * cutoff;
        alphaRed2 = alphaRed * alphaRed;
        setSelfEnergyPrefactor({-alphaRed * (1.0 + 0.5 * std::exp(-alphaRed2)) / pi_sqrt - 0.75 * std::erfc(alphaRed),
                                 -alphaRed * (2.0 * alphaRed2 * (1.0 / 3.0) + std::exp(-alphaRed2)) / pi_sqrt -
                                     0.5 * std::erfc(alphaRed)});
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
    inline void _to_json(nlohmann::json &j) const override {
        j = {{ "alpha", alpha }};
    }
#endif
};

// -------------- Wolf ---------------

/**
 * @brief Wolf scheme
 */
class Wolf : public EnergyImplementation<Wolf> {
  private:
    double alpha;               //!< Damping-parameter
    double alphaRed, alphaRed2; //!< Reduced damping-parameter, and squared
    const double pi_sqrt = 2.0 * std::sqrt(std::atan(1.0));
    const double pi = 4.0 * std::atan(1.0);

  public:
    /**
     * @param cutoff distance cutoff
     * @param alpha damping-parameter
     */
    inline Wolf(double cutoff, double alpha) : EnergyImplementation(Scheme::wolf, cutoff), alpha(alpha) {
        name = "Wolf";
        dipolar_selfenergy = true;
        doi = "10.1063/1.478738";
        alphaRed = alpha * cutoff;
        alphaRed2 = alphaRed * alphaRed;
        setSelfEnergyPrefactor({-alphaRed / pi_sqrt - std::erfc(alphaRed) / 2.0,
                                 -powi(alphaRed, 3) * 2.0 / 3.0 / pi_sqrt});
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
        chi = 2.0 * std::sqrt(pi) * cutoff * cutoff *
              (3.0 * std::exp(-alphaRed2) * alphaRed -
               std::sqrt(pi) * (std::erfc(alphaRed) * alphaRed2 + 3.0 * std::erf(alphaRed) * 0.5)) /
              (3.0 * alphaRed2);
    }

    inline double short_range_function(double q) const override {
        return (std::erfc(alphaRed * q) - q * std::erfc(alphaRed));
    }
    inline double short_range_function_derivative(double q) const override {
        return (-2.0 * std::exp(-alphaRed2 * q * q) * alphaRed / pi_sqrt - std::erfc(alphaRed));
    }
    inline double short_range_function_second_derivative(double q) const override {
        return 4.0 * std::exp(-alphaRed2 * q * q) * alphaRed2 * alphaRed * q / pi_sqrt;
    }
    inline double short_range_function_third_derivative(double q) const override {
        return -8.0 * std::exp(-alphaRed2 * q * q) * alphaRed2 * alphaRed * (alphaRed2 * q * q - 0.5) / pi_sqrt;
    }

#ifdef NLOHMANN_JSON_HPP
    /** Construct from JSON object, looking for `cutoff`, `alpha` */
    inline Wolf(const nlohmann::json &j) : Wolf(j.at("cutoff").get<double>(), j.at("alpha").get<double>()) {}

  private:
    inline void _to_json(nlohmann::json &j) const override {
        j = {{ "alpha", alpha }};
    }
#endif
};

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
        this->dipolar_selfenergy = true;
        this->doi = "10.1039/c9cp03875b";
        this->setSelfEnergyPrefactor({-0.5, -0.5});
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
    inline void _to_json(nlohmann::json &j) const override {
        j = {{ "order", order }};
    }
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
        dipolar_selfenergy = true;
        doi = "10.1039/c9cp03875b";
        setSelfEnergyPrefactor({-0.5, -0.5});
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
    inline void _to_json(nlohmann::json &j) const override {
        j = {{ "order", order }};
    }
#endif
};

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
    signed int C, D;            //!< Derivative cancelling-parameters
    double kappaRed, kappaRed2; //!< Debye-length
    double yukawa_denom, binomCDC;
    bool yukawa;

  public:
    /**
     * @param cutoff Spherical cutoff distance
     * @param C number of cancelled derivatives at origin -2 (starting from second derivative)
     * @param D number of cancelled derivatives at the cut-off (starting from zeroth derivative)
     * @param debye_length Debye screening length (infinite by default)
     */
    inline Poisson(double cutoff, signed int C, signed int D, double debye_length = infinity)
        : EnergyImplementation(Scheme::poisson, cutoff, debye_length), C(C), D(D) {
        if ( C < 1 )
            throw std::runtime_error("`C` must be larger than zero");
        if ( ( D < -1 ) && ( D != -C ) )
            throw std::runtime_error("If `D` is less than negative one, then it has to equal negative `C`");
        if ( ( D == 0 ) && ( C != 1 ) )
            throw std::runtime_error("If `D` is zero, then `C` has to equal one ");
        name = "poisson";
        dipolar_selfenergy = true;
        if(C < 2)
            dipolar_selfenergy = false;
        doi = "10/c5fr";
        double a1 = -double(C + D) / double(C);
        kappaRed = 0.0;
        yukawa = false;
        if( !std::isinf(debye_length) ) {
            kappaRed = cutoff / debye_length;
            if (std::fabs(kappaRed) > 1e-6) {
                yukawa = true;
                kappaRed2 = kappaRed * kappaRed;
                yukawa_denom = 1.0 / (1.0 - std::exp(2.0 * kappaRed));
                a1 *= -2.0 * kappaRed * yukawa_denom;
            }
        }
        binomCDC = 0.0;
        if( D != -C )
            binomCDC = double(binomial(C + D, C) * D);
        setSelfEnergyPrefactor({0.5 * a1, 0.0}); // Dipole self-energy seems to be 0 for C >= 2
        if (C == 2)
            setSelfEnergyPrefactor({0.5 * a1, -double(D) * (double(D * D) + 3.0 * double(D) + 2.0) / 12.0});
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) +
             short_range_function(0.0); // Is this OK for Yukawa-interactions?
        chi = -2.0 * std::acos(-1.0) * cutoff * cutoff * (1.0 + double(C)) * (2.0 + double(C)) /
              (3.0 * double(D + 1 + C) *
               double(D + 2 + C)); // not confirmed, but have worked for all tested values of 'C' and 'D'
    }

    inline double short_range_function(double q) const override {
        if( D == -C )
            return 1.0;
        double tmp = 0;
        double qp = q;
        if (yukawa)
            qp = (1.0 - std::exp(2.0 * kappaRed * q)) * yukawa_denom;
        if( ( D == 0 ) && ( C == 1 ) )
            return ( 1.0 - qp );
        for (signed int c = 0; c < C; c++)
            tmp += double(binomial(D - 1 + c, c)) * double(C - c) / double(C) * powi(qp, c);
        return powi(1.0 - qp, D + 1) * tmp;
    }

    inline double short_range_function_derivative(double q) const override {
        if( D == -C )
            return 0.0;
        if( ( D == 0 ) && ( C == 1 ) )
            return 0.0;
        double qp = q;
        double dqpdq = 1.0;
        if (yukawa) {
            double exp2kq = std::exp(2.0 * kappaRed * q);
            qp = (1.0 - exp2kq) * yukawa_denom;
            dqpdq = -2.0 * kappaRed * exp2kq * yukawa_denom;
        }
        double tmp1 = 1.0;
        double tmp2 = 0.0;
        for (int c = 1; c < C; c++) {
            double _fact = double(binomial(D - 1 + c, c)) * double(C - c) / double(C);
            tmp1 += _fact * powi(qp, c);
            tmp2 += _fact * double(c) * powi(qp, c - 1);
        }
        double dSdqp = (-double(D + 1) * powi(1.0 - qp, D) * tmp1 + powi(1.0 - qp, D + 1) * tmp2);
        return dSdqp * dqpdq;
    }

    inline double short_range_function_second_derivative(double q) const override {
        if( D == -C )
            return 0.0;
        if( ( D == 0 ) && ( C == 1 ) )
            return 0.0;
        double qp = q;
        double dqpdq = 1.0;
        double d2qpdq2 = 0.0;
        double dSdqp = 0.0;
        if (yukawa) {
            qp = (1.0 - std::exp(2.0 * kappaRed * q)) * yukawa_denom;
            dqpdq = -2.0 * kappaRed * std::exp(2.0 * kappaRed * q) * yukawa_denom;
            d2qpdq2 = -4.0 * kappaRed2 * std::exp(2.0 * kappaRed * q) * yukawa_denom;
            double tmp1 = 1.0;
            double tmp2 = 0.0;
            for (signed int c = 1; c < C; c++) {
                tmp1 += double(binomial(D - 1 + c, c)) * double(C - c) / double(C) * powi(qp, c);
                tmp2 += double(binomial(D - 1 + c, c)) * double(C - c) / double(C) * double(c) * powi(qp, c - 1);
            }
            dSdqp = (-double(D + 1) * powi(1.0 - qp, D) * tmp1 + powi(1.0 - qp, D + 1) * tmp2);
        }
        double d2Sdqp2 = binomCDC * powi(1.0 - qp, D - 1) * powi(qp, C - 1);
        return (d2Sdqp2 * dqpdq * dqpdq + dSdqp * d2qpdq2);
    };

    inline double short_range_function_third_derivative(double q) const override {
        if( D == -C )
            return 0.0;
        if( ( D == 0 ) && ( C == 1 ) )
            return 0.0;
        double qp = q;
        double dqpdq = 1.0;
        double d2qpdq2 = 0.0;
        double d3qpdq3 = 0.0;
        double d2Sdqp2 = 0.0;
        double dSdqp = 0.0;
        if (yukawa) {
            qp = (1.0 - std::exp(2.0 * kappaRed * q)) * yukawa_denom;
            dqpdq = -2.0 * kappaRed * std::exp(2.0 * kappaRed * q) * yukawa_denom;
            d2qpdq2 = -4.0 * kappaRed2 * std::exp(2.0 * kappaRed * q) * yukawa_denom;
            d3qpdq3 = -8.0 * kappaRed2 * kappaRed * std::exp(2.0 * kappaRed * q) * yukawa_denom;
            d2Sdqp2 = binomCDC * powi(1.0 - qp, D - 1) * powi(qp, C - 1);
            double tmp1 = 1.0;
            double tmp2 = 0.0;
            for (signed int c = 1; c < C; c++) {
                tmp1 += double(binomial(D - 1 + c, c)) * double(C - c) / double(C) * powi(qp, c);
                tmp2 += double(binomial(D - 1 + c, c)) * double(C - c) / double(C) * double(c) * powi(qp, c - 1);
            }
            dSdqp = (-double(D + 1) * powi(1.0 - qp, D) * tmp1 + powi(1.0 - qp, D + 1) * tmp2);
        }
        double d3Sdqp3 =
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
    inline void _to_json(nlohmann::json &j) const override {
        j = {{"C", C}, { "D", D }};
    }
#endif
};

// -------------- Fanourgakis ---------------

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
        dipolar_selfenergy = true;
        doi = "10.1063/1.3216520";
        setSelfEnergyPrefactor({-0.875, 0.0});
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
        chi = -5.0 * std::acos(-1.0) * cutoff * cutoff / 18.0;
    }

    inline double short_range_function(double q) const override {
        double q2 = q * q;
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

#ifdef NLOHMANN_JSON_HPP
inline std::shared_ptr<SchemeBase> createScheme(const nlohmann::json &j) {
    const std::map<std::string, Scheme> m = {{"plain", Scheme::plain},
                                             {"qpotential", Scheme::qpotential},
                                             {"wolf", Scheme::wolf},
                                             {"poisson", Scheme::poisson},
                                             {"reactionfield", Scheme::reactionfield},
                                             {"spline", Scheme::spline},
                                             {"fanourgakis", Scheme::fanourgakis},
                                             {"fennell", Scheme::fennell},
                                             {"zahn", Scheme::zahn},
                                             {"zerodipole", Scheme::zerodipole},
                                             {"ewald", Scheme::ewald},
                                             {"ewaldt", Scheme::ewaldt}}; // map string keyword to scheme type

    std::string name = j.at("type").get<std::string>();
    auto it = m.find(name);
    if (it == m.end())
        throw std::runtime_error("unknown coulomb scheme " + name);

    std::shared_ptr<SchemeBase> scheme;
    switch (it->second) {
    case Scheme::plain:
        scheme = std::make_shared<Plain>(j);
        break;
    case Scheme::wolf:
        scheme = std::make_shared<Wolf>(j);
        break;
    case Scheme::zahn:
        scheme = std::make_shared<Zahn>(j);
        break;
    case Scheme::fennell:
        scheme = std::make_shared<Fennell>(j);
        break;
    case Scheme::zerodipole:
        scheme = std::make_shared<ZeroDipole>(j);
        break;
    case Scheme::fanourgakis:
        scheme = std::make_shared<Fanourgakis>(j);
        break;
    case Scheme::qpotential5:
        scheme = std::make_shared<qPotentialFixedOrder<5>>(j);
        break;
    case Scheme::qpotential:
        scheme = std::make_shared<qPotential>(j);
        break;
    case Scheme::ewald:
        scheme = std::make_shared<Ewald>(j);
        break;
    case Scheme::ewaldt:
        scheme = std::make_shared<EwaldT>(j);
        break;
    case Scheme::poisson:
        scheme = std::make_shared<Poisson>(j);
        break;
    case Scheme::reactionfield:
        scheme = std::make_shared<ReactionField>(j);
        break;
    default:
        break;
    }
    return scheme;
}
#endif

// -------------- Splined ---------------

/**
 * @brief Dynamic scheme where all short ranged functions are splined
 *
 * This potential can hold any other scheme by splining all short-ranged
 * functions. To set to a particular scheme, call the `spline()` function
 * and pass the arguments required for the particular scheme.
 *
 * Example:
 *
 * ~~~{.cpp}
 *    Splined pot;
 *    double cutoff = 14;
 *    int order = 3;
 *    pot.spline<qPotential>(cutoff, order);
 * ~~~
 */
class Splined : public EnergyImplementation<Splined> {
  private:
    std::shared_ptr<SchemeBase> pot;
    Tabulate::Andrea<double> splined_srf;                            // spline class
    std::array<Tabulate::TabulatorBase<double>::data, 4> splinedata; // 0=original, 1=first derivative, ...

    inline void generate_spline_data() {
        assert(pot);
        SchemeBase::operator=(*pot); // copy base data from pot -> Splined
        splinedata[0] = splined_srf.generate([pot = pot](double q) { return pot->short_range_function(q); }, 0, 1);
        splinedata[1] =
            splined_srf.generate([pot = pot](double q) { return pot->short_range_function_derivative(q); }, 0, 1);
        splinedata[2] = splined_srf.generate(
            [pot = pot](double q) { return pot->short_range_function_second_derivative(q); }, 0, 1);
        splinedata[3] =
            splined_srf.generate([pot = pot](double q) { return pot->short_range_function_third_derivative(q); }, 0, 1);
    }

  public:
    inline Splined() : EnergyImplementation<Splined>(Scheme::spline, infinity) {
        setTolerance(1e-3);
    }

    /**
     * @brief Returns vector with number of spline knots the short-range-function and its derivatives
     */
    inline std::vector<size_t> numKnots() const {
        std::vector<size_t> n;
        for (auto &i : splinedata)
            n.push_back( i.numKnots() );
        return n;
    }

    /**
     * @brief Set relative spline tolerance
     */
    inline void setTolerance(double tol) {
        splined_srf.setTolerance(tol);
    }

    /**
     * @brief Spline given potential type
     * @tparam T Potential class
     * @param args Passed to constructor of potential class
     * @note This must be called before using any other functions
     */
    template <class T, class... Args> void spline(Args &&... args) {
        pot = std::make_shared<T>(args...);
        generate_spline_data();
    }
    inline double short_range_function(double q) const override { return splined_srf.eval(splinedata[0], q); };

    inline double short_range_function_derivative(double q) const override {
        return splined_srf.eval(splinedata[1], q);
    }
    inline double short_range_function_second_derivative(double q) const override {
        return splined_srf.eval(splinedata[2], q);
    }
    inline double short_range_function_third_derivative(double q) const override {
        return splined_srf.eval(splinedata[3], q);
    }
#ifdef NLOHMANN_JSON_HPP
  public:
    inline void to_json(nlohmann::json &j) const { pot->to_json(j); }

  private:
    inline void _to_json(nlohmann::json &) const override {}
#endif
};

} // namespace CoulombGalore
