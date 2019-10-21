#pragma once

#include <string>
#include <limits>
#include <cmath>
#include <iostream>
#include <Eigen/Core>

// https://en.cppreference.com/w/User:D41D8CD98F/feature_testing_macros#C.2B.2B17
#ifdef __cpp_lib_apply
#include <tuple>
using std::apply;
#elif __cpp_lib_exprimental_apply
#include <experimental/tuple>
using std::experimental::apply;
#else
//#error "no std::apply support ='("
#endif

namespace CoulombGalore {

typedef Eigen::Vector3d vec3; //!< typedef for 3d vector

constexpr double infinity = std::numeric_limits<double>::infinity(); //!< Numerical infinity

//!< Enum defining all possible schemes
enum class Scheme { plain, ewald, reactionfield, wolf, poisson, qpotential, fanourgakis, zahn, qpotential5, spline };

/**
 * @brief n'th integer power of float
 *
 * On GCC/Clang this will use the fast `__builtin_powi` function.
 *
 * See also:
 * - https://martin.ankerl.com/2012/01/25/optimized-approximative-pow-in-c-and-cpp
 * - https://martin.ankerl.com/2007/10/04/optimized-pow-approximation-for-java-and-c-c/
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

constexpr unsigned int binomial(unsigned int n, unsigned int k) {
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
 * where @f[ a=q^l @f]. In the implementation we use that
 * @f[
 *     (q^l;q)_P = (1-q)^P\prod_{n=1}^P\sum_{k=0}^{n+l}q^k
 * @f]
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
    };

    void setTolerance(T _utol, T _ftol = -1, T _umaxtol = -1, T _fmaxtol = -1) {
        utol = _utol;
        ftol = _ftol;
        umaxtol = _umaxtol;
        fmaxtol = _fmaxtol;
    }

    void setNumdr(T _numdr) { numdr = _numdr; }
};

/**
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

    /**
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
    /**
     * @brief Get tabulated value at f(x)
     * @param d Table data
     * @param r2 value
     */
    inline T eval(const typename base::data &d, T r2) const {
        size_t pos = std::lower_bound(d.r2.begin(), d.r2.end(), r2) - d.r2.begin() - 1;
        size_t pos6 = 6 * pos;
        assert((pos6 + 5) < d.c.size() && "out of bounds");
        T dz = r2 - d.r2[pos];
        return d.c[pos6] +
               dz * (d.c[pos6 + 1] +
                     dz * (d.c[pos6 + 2] + dz * (d.c[pos6 + 3] + dz * (d.c[pos6 + 4] + dz * (d.c[pos6 + 5])))));
    }

    /**
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
 */
class SchemeBase {
  protected:
    double invcutoff; // inverse cutoff distance
    double cutoff2;   // square cutoff distance
    double kappa;     // inverse Debye-length

  public:
    std::string doi;     //!< DOI for original citation
    std::string name;    //!< Descriptive name
    Scheme scheme;       //!< Truncation scheme
    double cutoff;       //!< Cut-off distance
    double debye_length; //!< Debye-length
    double T0; //!< Spatial Fourier transformed modified interaction tensor, used to calculate the dielectric constant
    std::array<double, 2> self_energy_prefactor; //!< Prefactor for self-energies
    inline SchemeBase(Scheme scheme, double cutoff, double debye_length = infinity)
        : scheme(scheme), cutoff(cutoff), debye_length(debye_length) {}

    virtual ~SchemeBase() = default;

    virtual double reciprocal_energy(std::vector<vec3>, std::vector<double>, std::vector<vec3>, vec3, int) const { return 0.0; }

    virtual double surface_energy(std::vector<vec3>, std::vector<double>, std::vector<vec3>, double) const { return 0.0; }

    virtual double charge_compensation_energy(std::vector<double>, double) const {return 0.0; }

    virtual vec3 reciprocal_force(std::vector<vec3>, std::vector<double>, std::vector<vec3>, int, vec3, int) const { return {0.0,0.0,0.0}; }

    virtual vec3 surface_force(std::vector<vec3>, std::vector<double>, std::vector<vec3>, int, double) const { return {0.0,0.0,0.0}; }

    virtual vec3 charge_compensation_force(std::vector<double>, vec3) const {return {0.0,0.0,0.0}; }

    /**
     * @brief Calculate dielectric constant
     * @param M2V see details
     *
     * @details The paramter @f[ M2V @f] is described by
     * @f[
     *     M2V = \frac{\langle M^2\rangle}{ 3\varepsilon_0Vk_BT }
     * @f]
     *
     * where @f[ \langle M^2\rangle @f] is mean value of the system dipole moment squared,
     * @f[ \varepsilon_0 @f] is the vacuum permittivity, @f[ V @f] the volume of the system,
     * @f[ k_B @f] the Boltzmann constant, @f[ T @f] the temperature, and @f[ T0 @f] the
     * Spatial Fourier transformed modified interaction tensor.
     */
    double calc_dielectric(double M2V) { return (M2V * T0 + 2.0 * M2V + 1.0) / (M2V * T0 - M2V + 1.0); }

    virtual double short_range_function(double q) const = 0;
    virtual double short_range_function_derivative(double q) const = 0;
    virtual double short_range_function_second_derivative(double q) const = 0;
    virtual double short_range_function_third_derivative(double q) const = 0;

    virtual double ion_ion_energy(double, double, double) const = 0;
    virtual double ion_dipole_energy(double, const vec3 &, const vec3 &) const = 0;
    virtual double dipole_dipole_energy(const vec3 &, const vec3 &, const vec3 &) const = 0;
    virtual double multipole_multipole_energy(double, double, const vec3 &, const vec3 &, const vec3 &) const = 0;

    virtual vec3 ion_ion_force(double, double, const vec3 &) const = 0;
    virtual vec3 ion_dipole_force(double, const vec3 &, const vec3 &) const = 0;
    virtual vec3 dipole_dipole_force(const vec3 &, const vec3 &, const vec3 &) const = 0;
    virtual vec3 multipole_multipole_force(double, double, const vec3 &, const vec3 &, const vec3 &) const = 0;

    virtual double ion_potential(double, double) const = 0;
    virtual double dipole_potential(const vec3 &, const vec3 &) const = 0;

    // add remaining funtions here...

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
template <class T, bool debyehuckel=true> class EnergyImplementation : public SchemeBase {
  public:
    EnergyImplementation(Scheme type, double cutoff, double debyelength = infinity)
        : SchemeBase(type, cutoff, debyelength) {
        invcutoff = 1.0 / cutoff;
        cutoff2 = cutoff * cutoff;
        kappa = 1.0 / debyelength;
    }

    /**
     * @brief potential from ion
     * @returns potential from ion in electrostatic units ( why not Hartree atomic units? )
     * @param z charge
     * @param r distance from charge
     *
     * @details The potential from a charge is described by
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
     * @brief potential from dipole
     * @returns potential from dipole in electrostatic units ( why not Hartree atomic units? )
     * @param mu dipole
     * @param r distance-vector from dipole
     *
     * @details The potential from a charge is described by
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
            return mu.dot(r) / (r2 * r1) *
                   (static_cast<const T *>(this)->short_range_function(q) * (1.0 + kappa * r1) -
                    q * static_cast<const T *>(this)->short_range_function_derivative(q)) *
                   std::exp(-kappa * r1);
        } else {
            return 0.0;
        }
    }

    /**
     * @brief field from ion
     * @returns field in electrostatic units ( why not Hartree atomic units? )
     * @param z charge
     * @param r distance-vector from charge
     *
     * @details The field from a charge is described by
     * @f[
     *     {\bf E}(z, {\bf r}) = \frac{z \hat{{\bf r}} }{|{\bf r}|^2} \left( s(q) - qs^{\prime}(q) \right)
     * @f]
     */
    inline vec3 ion_field(double z, const vec3 &r) const {
        double r2 = r.squaredNorm();
        if (r2 < cutoff2) {
            double r1 = std::sqrt(r2);
            double q = r1 * invcutoff;
            return z * r / (r2 * r1) *
                   (static_cast<const T *>(this)->short_range_function(q) * (1.0 + kappa * r1) -
                    q * static_cast<const T *>(this)->short_range_function_derivative(q)) *
                   std::exp(-kappa * r1);
        } else {
            return {0, 0, 0};
        }
    }

    /**
     * @brief field from dipole
     * @returns field in electrostatic units ( why not Hartree atomic units? )
     * @param mu dipole
     * @param r distance-vector from dipole
     *
     * @details The field from a dipole is described by
     * @f[
     *     {\bf E}(\boldsymbol{\mu}, {\bf r}) = \frac{3 ( \boldsymbol{\mu} \cdot \hat{{\bf r}} ) \hat{{\bf r}} -
     * \boldsymbol{\mu} }{|{\bf r}|^3} \left( s(q) - qs^{\prime}(q)  + \frac{q^2}{3}s^{\prime\prime}(q) \right) +
     * \frac{\boldsymbol{\mu}}{|{\bf r}|^3}\frac{q^2}{3}s^{\prime\prime}(q)
     * @f]
     */
    inline vec3 dipole_field(const vec3 &mu, const vec3 &r) const {
        double r2 = r.squaredNorm();
        if (r2 < cutoff2) {
            double r1 = std::sqrt(r2);
            double r3 = r1 * r2;
            double q = r1 * invcutoff;
            double q2 = q * q;
            double kappa2 = kappa * kappa;
            double kappa_x_r1 = kappa * r1;
            double srf = static_cast<const T *>(this)->short_range_function(q);
            double dsrf = static_cast<const T *>(this)->short_range_function_derivative(q);
            double ddsrf = static_cast<const T *>(this)->short_range_function_second_derivative(q);
            vec3 fieldD = (3.0 * mu.dot(r) * r / r2 - mu) / r3;
            fieldD *= (srf * (1.0 + kappa_x_r1 + kappa2 * r2 / 3.0) - q * dsrf * (1.0 + 2.0 / 3.0 * kappa_x_r1) +
                       q2 / 3.0 * ddsrf);
            vec3 fieldI = mu / r3 * (srf * kappa2 * r2 - 2.0 * kappa_x_r1 * q * dsrf + ddsrf * q2) / 3.0;
            return (fieldD + fieldI) * std::exp(-kappa_x_r1);
        } else {
            return {0, 0, 0};
        }
    }

    /**
     * @brief interaction energy between two ions
     * @returns interaction energy in electrostatic units ( why not Hartree atomic units? )
     * @param zA charge
     * @param zB charge
     * @param r charge separation
     *
     * @details The interaction energy between two charges is decribed by
     * @f[
     *     u(z_A, z_B, r) = z_B \Phi(z_A,r)
     * @f]
     * where @f[ \Phi(z_A,r) @f] is the potential from ion A.
     */
    inline double ion_ion_energy(double zA, double zB, double r) const override { return zB * ion_potential(zA, r); }

    /**
     * @brief interaction energy between an ion and a dipole
     * @returns interaction energy in electrostatic units ( why not Hartree atomic units? )
     * @param z charge
     * @param mu dipole moment
     * @param r distance-vector between dipole and charge, @f[ {\bf r} = {\bf r}_{\mu} - {\bf r}_z @f]
     *
     * @details The interaction energy between an ion and a dipole is decribed by
     * @f[
     *     u(z, \boldsymbol{\mu}, {\bf r}) = z \Phi(\boldsymbol{\mu}, -{\bf r})
     * @f]
     * where @f[ \Phi(\boldsymbol{\mu}, -{\bf r}) @f] is the potential from the dipole at the location of the ion.
     * This interaction can also be described by
     * @f[
     *     u(z, \boldsymbol{\mu}, {\bf r}) = -\boldsymbol{\mu}\cdot {\bf E}(z, {\bf r})
     * @f]
     * where @f[ {\bf E}(z, {\bf r}) @f] is the field from the ion at the location of the dipole.
     */
    inline double ion_dipole_energy(double z, const vec3 &mu, const vec3 &r) const override {
        // Both expressions below gives same answer. Keep for possible optimization in future.
        // return -mu.dot(ion_field(z,r)); // field from charge interacting with dipole
        return z * dipole_potential(mu, -r); // potential of dipole interacting with charge
    }

    /**
     * @brief interaction energy between two dipoles
     * @returns interaction energy in electrostatic units ( why not Hartree atomic units? )
     * @param muA dipole moment of particle A
     * @param muB dipole moment of particle B
     * @param r distance-vector between dipoles, @f[ {\bf r} = {\bf r}_{\mu_B} - {\bf r}_{\mu_A} @f]
     *
     * @details The interaction energy between two dipoles is decribed by
     * @f[
     *     u(\boldsymbol{\mu}_A, \boldsymbol{\mu}_B, {\bf r}) = -\boldsymbol{\mu}_A\cdot {\bf E}(\boldsymbol{\mu}_B,
     * {\bf r})
     * @f]
     * where @f[ {\bf E}(\boldsymbol{\mu}_B, {\bf r}) @f] is the field from dipole B at the location of dipole A.
     */
    inline double dipole_dipole_energy(const vec3 &muA, const vec3 &muB, const vec3 &r) const override {
        return -muA.dot(dipole_field(muB, r));
    }

    /**
     * @brief interaction energy between two multipoles with charges and dipole moments
     * @returns interaction energy in electrostatic units ( why not Hartree atomic units? )
     * @param zA charge of particle A
     * @param zB charge of particle B
     * @param muA dipole moment of particle A
     * @param muB dipole moment of particle B
     * @param r distance-vector between dipoles, @f[ {\bf r} = {\bf r}_{\mu_B} - {\bf r}_{\mu_A} @f]
     *
     * @details A combination of the functions 'ion_ion_energy', 'ion_dipole_energy' and 'dipole_dipole_energy'.
     */
    inline double multipole_multipole_energy(double zA, double zB, const vec3 &muA, const vec3 &muB, const vec3 &r) const override {
        double r2 = r.squaredNorm();
        if (r2 < cutoff2) {
            double r1 = std::sqrt( r2 );
            double r3 = r1 * r2;
            double q = r1 / cutoff;
            double kappa_r1 = kappa * r1;

            double srf = static_cast<const T *>(this)->short_range_function(q);
            double dsrfq = static_cast<const T *>(this)->short_range_function_derivative(q) * q;
            double ddsrfq2 = static_cast<const T *>(this)->short_range_function_second_derivative(q) * q * q / 3.0;

            double tmp1 = (srf * (1.0 + kappa_r1) - dsrfq) / r3;
            double tmp2 = ( srf * kappa_r1 * kappa_r1 / 3.0 - dsrfq * (2.0 / 3.0 * kappa_r1) + ddsrfq2 ) / r3;

            vec3 field_dipoleB = (3.0 * muB.dot(r) * r / r2 - muB) * ( tmp1 + tmp2 ) + muB * tmp2;

            double ion_ion = zA * zB / r1 * srf;
            double ion_dipole = ( zB * muA.dot(r) + zA * muB.dot(-r) ) * tmp1;
            double dipole_dipole = -muA.dot(field_dipoleB);
            return ( ion_ion + ion_dipole + dipole_dipole ) * std::exp(-kappa_r1);
        } else {
            return 0.0;
        }
    }

    /**
     * @brief ion-ion interaction force
     * @returns interaction force in electrostatic units ( why not Hartree atomic units? )
     * @param zA charge
     * @param zB charge
     * @param r distance-vector between charges, @f[ {\bf r} = {\bf r}_{z_B} - {\bf r}_{z_A} @f]
     *
     * @details The force between two ions is decribed by
     * @f[
     *     {\bf F}(z_A, z_B, {\bf r}) = z_B {\bf E}(z_A, {\bf r})
     * @f]
     * where @f[ {\bf E}(z_A, {\bf r}) @f] is the field from ion A at the location of ion B.
     */
    inline vec3 ion_ion_force(double zA, double zB, const vec3 &r) const override { return zB * ion_field(zA, r); }

    /**
     * @brief ion-dipole interaction force
     * @returns interaction force in electrostatic units ( why not Hartree atomic units? )
     * @param z charge
     * @param mu dipole moment
     * @param r distance-vector between dipole and charge, @f[ {\bf r} = {\bf r}_{\mu} - {\bf r}_z @f]
     *
     * @details The force between an ion and a dipole is decribed by
     * @f[
     *     {\bf F}(z, \boldsymbol{\mu}, {\bf r}) = z {\bf E}(\boldsymbol{\mu}, {\bf r})
     * @f]
     * where @f[ {\bf E}(\boldsymbol{\mu}, {\bf r}) @f] is the field from the dipole at the location of the ion.
     */
    inline vec3 ion_dipole_force(double z, const vec3 &mu, const vec3 &r) const override {
        return z * dipole_field(mu, r);
    }

    /**
     * @brief dipole-dipole interaction force
     * @returns interaction force in electrostatic units ( why not Hartree atomic units? )
     * @param muA dipole moment of particle A
     * @param muB dipole moment of particle B
     * @param r distance-vector between dipoles, @f[ {\bf r} = {\bf r}_{\mu_B} - {\bf r}_{\mu_A} @f]
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
            double r4 = r2 * r2;
            double muAdotRh = muA.dot(rh);
            double muBdotRh = muB.dot(rh);
            vec3 forceD =
                3.0 * ((5.0 * muAdotRh * muBdotRh - muA.dot(muB)) * rh - muBdotRh * muA - muAdotRh * muB) / r4;
            double srf = static_cast<const T *>(this)->short_range_function(q);
            double dsrf = static_cast<const T *>(this)->short_range_function_derivative(q);
            double ddsrf = static_cast<const T *>(this)->short_range_function_second_derivative(q);
            double dddsrf = static_cast<const T *>(this)->short_range_function_third_derivative(q);
            forceD *= (srf * (1.0 + kappa * r1 + kappa * kappa * r2 / 3.0) - q * dsrf * (1.0 + 2.0 / 3.0 * kappa * r1) +
                       q2 / 3.0 * ddsrf);
            vec3 forceI = muAdotRh * muBdotRh * rh / r4;
            forceI *=
                (srf * (1.0 + kappa * r1) * kappa * kappa * r2 - q * dsrf * (3.0 * kappa * r1 + 2.0) * kappa * r1 +
                 ddsrf * (1.0 + 3.0 * kappa * r1) * q2 - q2 * q * dddsrf);
            return (forceD + forceI) * std::exp(-kappa * r1);
        } else {
            return {0, 0, 0};
        }
    }

    /**
     * @brief interaction force between two multipoles with charges and dipole moments
     * @returns interaction force in electrostatic units ( why not Hartree atomic units? )
     * @param zA charge of particle A
     * @param zB charge of particle B
     * @param muA dipole moment of particle A
     * @param muB dipole moment of particle B
     * @param r distance-vector between dipoles, @f[ {\bf r} = {\bf r}_{\mu_B} - {\bf r}_{\mu_A} @f]
     *
     * @details A combination of the functions 'ion_ion_force', 'ion_dipole_force' and 'dipole_dipole_force'.
     * @warning Not working!
     */
    inline vec3 multipole_multipole_force(double zA, double zB, const vec3 &muA, const vec3 &muB, const vec3 &r) const override {
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
            double ddsrfq2 = static_cast<const T *>(this)->short_range_function_second_derivative(q) * q2;
            double dddsrfq3 = static_cast<const T *>(this)->short_range_function_third_derivative(q) * q2 * q;

	    double tmp0 = (srf * (1.0 + kr) - dsrfq);
	    double tmp1 = ( srf * kr * kr - 2.0 * kr * dsrfq + ddsrfq2 ) / 3.0;
	    double tmp2 = tmp1 + tmp0;
	    double tmp3 = (srf * (1.0 + kr) * kr * kr - dsrfq * (2.0 * ( 1.0 + kr) + kr) * kr + ddsrfq2 * (1.0 + 3.0 * kr) - dddsrfq3) / r1;
            
	    
            vec3 ion_ion = zB * zA * r * (srf * (1.0 + kr) - dsrfq);
	    //vec3 ion_ion = zB * zA * r * tmp0;
	    
            vec3 ion_dipoleA = zA * ( (3.0 * muBdotRh * rh - muB) * tmp2 + muB * tmp1)*0.0;
            vec3 ion_dipoleB = zB * ( (3.0 * muAdotRh * rh - muA) * tmp2 + muA * tmp1)*0.0;

            vec3 forceD = 3.0 * ((5.0 * muAdotRh * muBdotRh - muA.dot(muB)) * rh - muBdotRh * muA - muAdotRh * muB) * tmp2 / r1;
            vec3 dipole_dipole = ( forceD +  muAdotRh * muBdotRh * rh * tmp3 );
	    return ( ion_ion + ion_dipoleA + ion_dipoleB + dipole_dipole ) * std::exp(-kr) / r2 / r1;
        } else {
            return {0, 0, 0};
        }
    }

    /**
     * @brief torque exerted on dipole
     * @returns torque on dipole in electrostatic units ( why not Hartree atomic units? )
     * @param mu dipole moment
     * @param E field
     *
     * @details The torque on a dipole in a field is described by
     * @f[
     *     \boldsymbol{\tau} = \boldsymbol{\mu} \times \boldsymbol{E}
     * @f]
     */
    inline vec3 dipole_torque(const vec3 &mu, const vec3 &E) const { return mu.cross(E); }

    /**
     * @brief self-energy for all type of interactions
     * @returns self energy in electrostatic units ( why not Hartree atomic units? )
     * @param m2 vector with square moments, \textit{i.e.} charge squared, dipole moment squared, etc.
     *
     * @details The torque on a dipole in a field is described by
     * @f[
     *     u_{self} = p_1 \frac{z^2}{R_c} + p_2 \frac{|\boldsymbol{\mu}|^2}{R_c^3} + \cdots
     * @f]
     * where @f[ p_i @f] is the prefactor for the self-energy for species 'i'.
     * Here i=0 represent ions, i=1 represent dipoles etc.
     */
    inline double self_energy(const std::array<double, 2> &m2) const {
        static_assert(decltype(m2){0}.size() == decltype(self_energy_prefactor){0}.size(), "static assert");
        double e_self = 0.0;
        for (int i = 0; i < (int)m2.size(); i++)
            e_self += self_energy_prefactor[i] * m2[i] * powi(invcutoff, 2 * i + 1);
        return e_self;
    }
};

// -------------- Plain ---------------

/**
 * @brief No truncation scheme, cutoff = infinity
 */
class Plain : public EnergyImplementation<Plain> {
  public:
    inline Plain(double debye_length = infinity) : EnergyImplementation(Scheme::plain, std::numeric_limits<double>::max(), debye_length) {
        name = "plain";
        doi = "Premier mémoire sur l’électricité et le magnétisme by Charles-Augustin de Coulomb"; // :P
        self_energy_prefactor = {0.0, 0.0};
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
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

// -------------- Ewald real-space ---------------

/**
 * @brief Ewald real-space scheme
 */
struct Ewald : public EnergyImplementation<Ewald> {
    double alpha, alpha2;                  //!< Damping-parameter
    double alphaRed, alphaRed2, alphaRed3; //!< Reduced damping-parameter, and squared
    double eps_sur;                        //!< Dielectric constant of the surrounding medium
    double debye_length;                   //!< Debye-length
    double kappa, kappa2;                  //!< Inverse Debye-length
    double beta, beta2, beta3;             //!< Inverse ( twice Debye-length times damping-parameter )
    const double pi_sqrt = 2.0 * std::sqrt(std::atan(1.0));
    const double pi = 4.0 * std::atan(1.0);

  public:
    /**
     * @param cutoff distance cutoff
     * @param alpha damping-parameter
     */
    inline Ewald(double cutoff, double alpha, double eps_sur = infinity, double debye_length = infinity)
        : EnergyImplementation(Scheme::ewald, cutoff), alpha(alpha), eps_sur(eps_sur) {
        name = "Ewald real-space";
        alpha2 = alpha * alpha;
        alphaRed = alpha * cutoff;
        alphaRed2 = alphaRed * alphaRed;
        alphaRed3 = alphaRed2 * alphaRed;
        if (eps_sur < 1.0)
            eps_sur = infinity;
        T0 = (std::isinf(eps_sur)) ? 1.0 : 2.0 * (eps_sur - 1.0) / (2.0 * eps_sur + 1.0);
        kappa = 1.0 / debye_length;
        kappa2 = kappa * kappa;
        beta = kappa / (2.0 * alpha);
        beta2 = beta * beta;
        beta3 = beta2 * beta;
        self_energy_prefactor = {
            -alphaRed / pi_sqrt * (std::exp(-beta2) - pi_sqrt * beta * std::erfc(beta)),
            -alphaRed3 * 2.0 / 3.0 / pi_sqrt *
                (2.0 * pi_sqrt * beta3 * std::erfc(beta) + (1.0 - 2.0 * beta2) * std::exp(-beta2))};
    }

    inline double short_range_function(double q) const override {
        return 0.5 *
               (std::erfc(alphaRed * q + beta) * std::exp(4.0 * alphaRed * beta * q) + std::erfc(alphaRed * q - beta));
    }
    inline double short_range_function_derivative(double q) const override {
        double expC = std::exp(-powi(alphaRed * q - beta, 2));
        double erfcC = std::erfc(alphaRed * q + beta);
        return (-2.0 * alphaRed / pi_sqrt * expC + 2.0 * alphaRed * beta * erfcC * std::exp(4.0 * alphaRed * beta * q));
    }
    inline double short_range_function_second_derivative(double q) const override {
        double expC = std::exp(-powi(alphaRed * q - beta, 2));
        double erfcC = std::erfc(alphaRed * q + beta);
        return (4.0 * alphaRed2 / pi_sqrt * (alphaRed * q - 2.0 * beta) * expC +
                8.0 * alphaRed2 * beta2 * erfcC * std::exp(4.0 * alphaRed * beta * q));
    }
    inline double short_range_function_third_derivative(double q) const override {
        double expC = std::exp(-powi(alphaRed * q - beta, 2));
        double erfcC = std::erfc(alphaRed * q + beta);
        return (4.0 * alphaRed3 / pi_sqrt *
                    (1.0 - 2.0 * (alphaRed * q - 2.0 * beta) * (alphaRed * q - beta) - 4.0 * beta2) * expC +
                32.0 * alphaRed3 * beta3 * erfcC * std::exp(4.0 * alphaRed * beta * q));
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
    inline double reciprocal_energy(std::vector<vec3> positions, std::vector<double> charges, std::vector<vec3> dipoles, vec3 L, int nmax) const {
        double volume = L[0]*L[1]*L[2];
        if( std::abs( int(positions.size()) - int(charges.size()) ) > 0 ||
                std::abs( int(positions.size()) - int(dipoles.size()) ) > 0 ||
                std::abs( int(charges.size()) - int(dipoles.size()) ) > 0 )
            throw std::runtime_error("Vectors must have same size!");

        std::vector<vec3> kvec;
        std::vector<double> Ak;
        int kvec_size = 0;
        for(int nx = -nmax; nx < nmax+1; nx++) {
            for(int ny = -nmax; ny < nmax+1; ny++) {
                for(int nz = -nmax; nz < nmax+1; nz++) {
                    vec3 kv = { 2.0 * pi * nx / L[0] , 2.0 * pi * ny / L[1] , 2.0 * pi * nz / L[2] };
                    double k2 = double( kv.squaredNorm() ) + kappa2;
                    vec3 nv = { double(nx) , double(ny) , double(nz) };
		    double nv1 = double( nv.norm() );
                    if( nv1 > 0 && nv1 <= nmax) {
                        kvec.push_back(kv);
                        Ak.push_back( std::exp( -k2 / 4.0 / alpha2 - beta2 ) / k2 );
                        kvec_size++;
                    }
                }
            }
        }

        double E = 0.0;
        for(int k = 0; k < kvec_size; k++) {
            std::complex<double> Qq(0.0,0.0);
            std::complex<double> Qmu(0.0,0.0);
            for(unsigned int i = 0; i < positions.size(); i++) {
                double kDotR = kvec.at(k).dot( positions.at(i) );
                Qq += charges.at(i) * std::complex<double>( cos(kDotR) , sin(kDotR) );
                Qmu += dipoles.at(i).dot(kvec.at(k)) * std::complex<double>( -sin(kDotR) , cos(kDotR) );
            }
            std::complex<double> Q = Qq + Qmu;
            E += ( pow( std::abs( Q ), 2.0 ) * Ak.at(k) );
        }
        return ( E * 2.0 * pi / volume );
    }

    /**
     * @brief Surface-term
     * @param positions Positions of particles
     * @param charges Charges of particles
     * @param dipoles Dipole moments of particles
     * @param volume Volume of unit-cell
     */
    inline double surface_energy(std::vector<vec3> positions, std::vector<double> charges, std::vector<vec3> dipoles, double volume) const {
        if( std::abs( int(positions.size()) - int(charges.size()) ) > 0 ||
                std::abs( int(positions.size()) - int(dipoles.size()) ) > 0 ||
                std::abs( int(charges.size()) - int(dipoles.size()) ) > 0 )
            throw std::runtime_error("Vectors must have same size!");

        vec3 sum_r_charges = {0.0,0.0,0.0};
        vec3 sum_dipoles = {0.0,0.0,0.0};
        for(unsigned int i = 0; i < positions.size(); i++) {
            sum_r_charges += positions.at(i) * charges.at(i);
            sum_dipoles += dipoles.at(i);
        }
        double sqDipoles = sum_r_charges.dot(sum_r_charges) + 2.0 * sum_r_charges.dot(sum_dipoles) + sum_dipoles.dot(sum_dipoles);
        return ( 2.0 * pi / ( 2.0 * eps_sur + 1.0 ) / volume * sqDipoles );
    }

    inline vec3 surface_force(std::vector<vec3> positions, std::vector<double> charges, std::vector<vec3> dipoles, int I, double volume) const {
        if( std::abs( int(positions.size()) - int(charges.size()) ) > 0 ||
                std::abs( int(positions.size()) - int(dipoles.size()) ) > 0 ||
                std::abs( int(charges.size()) - int(dipoles.size()) ) > 0 )
            throw std::runtime_error("Vectors must have same size!");

        vec3 sum_r_charges = {0.0,0.0,0.0};
        vec3 sum_dipoles = {0.0,0.0,0.0};
        for(unsigned int i = 0; i < positions.size(); i++) {
            sum_r_charges += positions.at(i) * charges.at(i);
            sum_dipoles += dipoles.at(i);
        }
        double sqDipoles = 0.0;
        return ( -4.0*pi/(2.0*eps_sur + 1.0)/volume * charges.at(I) *(sum_r_charges + sum_dipoles) );
    }

    /**
     * @brief Compensating term for non-neutral systems
     * @param charges Charges of particles
     * @param volume Volume of unit-cell
     * @note DOI:10.1021/ct400626b
     */
    inline double charge_compensation_energy(std::vector<double> charges, double volume) const {
        double squaredSumQ = 0.0;
        for(unsigned int i = 0; i < charges.size(); i++)
            squaredSumQ += charges.at(i);
        return ( -pi / 2.0 / alpha2 / volume * squaredSumQ );
    }

#ifdef NLOHMANN_JSON_HPP
    inline Ewald(const nlohmann::json &j)
        : Ewald(j.at("cutoff").get<double>(), j.at("alpha").get<double>(), j.value("epss", infinity),
                j.value("debyelength", infinity)) {}

  private:
    inline void _to_json(nlohmann::json &j) const override {
        j = {{ "alpha", alpha }};
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
    double epsRF;               //!< Relative permittivity of the surrounding medium
    double epsr;                //!< Relative permittivity of the dispersing medium
    bool shifted;               //!< Shifted to zero potential at the cut-off

  public:
    /**
     * @param cutoff distance cutoff
     * @param epsRF dielectric constant of the surrounding
     * @param epsr dielectric constant of the sample
     * @param shifted shifted potential
     */
    inline ReactionField(double cutoff, double epsRF, double epsr, bool shifted) : EnergyImplementation(Scheme::reactionfield, cutoff), epsRF(epsRF), epsr(epsr), shifted(shifted) {
        name = "Reaction-field";
        epsRF = epsRF;
        epsr = epsr;
	shifted = shifted;
        self_energy_prefactor = { -3.0 * epsRF * double(shifted) / ( 4.0 * epsRF + 2.0 * epsr ), 0.0};
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
    }

    inline double short_range_function(double q) const override {
        return ( 1.0 + ( epsRF - epsr ) * q * q * q / ( 2.0 * epsRF + epsr ) - 3.0 * epsRF * q / ( 2.0 * epsRF + epsr ) * double(shifted) );
    }
    inline double short_range_function_derivative(double q) const override {
        return ( 3.0 * ( epsRF - epsr ) * q * q / ( 2.0 * epsRF + epsr ) - 3.0 * epsRF * double(shifted) / ( 2.0 * epsRF + epsr ) );
    }
    inline double short_range_function_second_derivative(double q) const override {
        return 6.0 * ( epsRF - epsr ) * q / ( 2.0 * epsRF + epsr );
    }
    inline double short_range_function_third_derivative(double q) const override {
        return 6.0 * ( epsRF - epsr ) / ( 2.0 * epsRF + epsr );
    }

#ifdef NLOHMANN_JSON_HPP
    inline ReactionField(const nlohmann::json &j) : ReactionField(j.at("cutoff").get<double>(), j.at("epsRF").get<double>(), j.at("epsr").get<double>(), j.at("shifted").get<bool>()) {}

  private:
    inline void _to_json(nlohmann::json &j) const override {
        j = {{"epsr", epsr}, { "epsRF", epsRF }, { "shifted", shifted }};
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

  public:
    /**
     * @param cutoff distance cutoff
     * @param alpha damping-parameter
     */
    inline Zahn(double cutoff, double alpha) : EnergyImplementation(Scheme::zahn, cutoff), alpha(alpha) {
        name = "Zahn";
        alphaRed = alpha * cutoff;
        alphaRed2 = alphaRed * alphaRed;
        self_energy_prefactor = { -alphaRed * ( 1.0 - std::exp(-alphaRed2) ) / pi_sqrt + 0.5 * std::erfc(alphaRed), 0.0};
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
    }

    inline double short_range_function(double q) const override {
        return ( std::erfc(alphaRed * q) - ( q - 1.0 ) * q * ( std::erfc( alphaRed ) + 2.0 * alphaRed * std::exp( -alphaRed2 ) / pi_sqrt ) );
    }//          std::erfc(alphaRed * q) - ( q - 1.0 ) * q * ( std::erfc( alphaRed ) + 2.0 * alphaRed * std::exp( -alphaRed2 ) / pi_sqrt )
    inline double short_range_function_derivative(double q) const override {
        return ( -( 4.0 * ( 0.5 * std::exp(-alphaRed2*q*q) * alphaRed + ( alphaRed * std::exp(-alphaRed2) + 0.5 * pi_sqrt * std::erfc(alphaRed) ) * ( q - 0.5 ) ) ) / pi_sqrt );
    }
    inline double short_range_function_second_derivative(double q) const override {
        return ( 4.0 * ( alphaRed2*alphaRed * q * std::exp( -alphaRed2 * q * q ) - alphaRed * std::exp( -alphaRed2 ) - 0.5 * pi_sqrt * std::erfc(alphaRed) ) ) / pi_sqrt;
    }
    inline double short_range_function_third_derivative(double q) const override {
        return ( -8.0 * std::exp( -alphaRed2 * q * q ) * ( alphaRed2 * q*q - 0.5 ) * alphaRed2*alphaRed / pi_sqrt );
    }

#ifdef NLOHMANN_JSON_HPP
    inline Zahn(const nlohmann::json &j) : Zahn(j.at("cutoff").get<double>(), j.at("alpha").get<double>()) {}

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

  public:
    /**
     * @param cutoff distance cutoff
     * @param alpha damping-parameter
     */
    inline Wolf(double cutoff, double alpha) : EnergyImplementation(Scheme::wolf, cutoff), alpha(alpha) {
        name = "Wolf";
        alphaRed = alpha * cutoff;
        alphaRed2 = alphaRed * alphaRed;
        self_energy_prefactor = {-alphaRed / pi_sqrt - std::erfc(alphaRed)/2.0, -powi(alphaRed, 3) * 2.0 / 3.0 / pi_sqrt};
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
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
    using base::name;
    using base::self_energy_prefactor;
    using base::T0;
    /**
     * @param cutoff distance cutoff
     * @param order number of moments to cancel
     */
    inline qPotentialFixedOrder(double cutoff) : base(Scheme::qpotential, cutoff) {
        name = "qpotential";
        this->doi = "10/c5fr";
        self_energy_prefactor = {-0.5, -0.5};
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
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
        doi = "10/c5fr";
        self_energy_prefactor = {-0.5, -0.5};
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
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
    inline qPotential(const nlohmann::json &j)
        : qPotential(j.at("cutoff").get<double>(), j.at("order").get<double>()) {}

  private:
    inline void _to_json(nlohmann::json &j) const override {
        j = {{ "order", order }};
    }
#endif
};

/**
 * @brief Poisson scheme, also works for Yukawa-potential
 *
 * A general scheme which pending two parameters `C` and `D` can model several different pair-potentials.
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
 *  The following keywords are required:
 *
 *  Keyword        |  Description
 *  -------------- |  -------------------------------------------
 *  `cutoff`       |  Spherical cutoff
 *  `C`            |  Number of cancelled derivatives at origin -2 (starting from second derivative)
 *  `D`            |  Number of cancelled derivatives at the cut-off (starting from zeroth derivative)
 *  `debye_length` |  Debye-length (optional)
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
    inline Poisson(double cutoff, signed int C, signed int D, double debye_length = infinity)
        : EnergyImplementation(Scheme::poisson, cutoff, debye_length), C(C), D(D) {
        if ((C < 1) || (D < -1))
            throw std::runtime_error("`C` must be larger than zero and `D` must be larger or equal to negative one");
        name = "poisson";
        doi = "10/c5fr";
        double a1 = -double(C + D) / double(C);
        kappaRed = cutoff / debye_length;
        yukawa = false;
        if (std::fabs(kappaRed) > 1e-6) {
            yukawa = true;
            kappaRed2 = kappaRed * kappaRed;
            yukawa_denom = 1.0 / (1.0 - std::exp(2.0 * kappaRed));
            a1 *= -2.0 * kappaRed * yukawa_denom;
        }
        binomCDC = double(binomial(C + D, C) * D);
        self_energy_prefactor = {0.5*a1, 0.0};
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) +
             short_range_function(0.0); // Is this OK for Yukawa-interactions?
    }

    inline double short_range_function(double q) const override {
        double tmp = 0;
        double qp = q;
        if (yukawa)
            qp = (1.0 - std::exp(2.0 * kappaRed * q)) * yukawa_denom;
        for (signed int c = 0; c < C; c++)
            tmp += double(binomial(D - 1 + c, c)) * double(C - c) / double(C) * powi(qp, c);
        return powi(1.0 - qp, D + 1) * tmp;
    }

    inline double short_range_function_derivative(double q) const override {
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
        doi = "10.1063/1.3216520";
        self_energy_prefactor = {-0.875, 0.0};
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
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
                                             {"spline", Scheme::spline},
                                             {"fanourgakis", Scheme::fanourgakis},
                                             {"ewald", Scheme::ewald}}; // map string keyword to scheme type

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
    case Scheme::poisson:
        scheme = std::make_shared<Poisson>(j);
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
 * @tparam spline If false, no splining if performed
 */
class Splined : public EnergyImplementation<Splined> {
  private:
    std::shared_ptr<SchemeBase> pot;
    Tabulate::Andrea<double> splined_srf;                            // spline class
    std::array<Tabulate::TabulatorBase<double>::data, 4> splinedata; // 0=original, 1=first derivative, ...

    void generate_spline_data() {
        assert(pot);
        SchemeBase::operator=(*pot); // copy base data from pot -> Splined
        splined_srf.setTolerance(1e-3, 1e-1);
        splinedata[0] = splined_srf.generate([pot = pot](double q) { return pot->short_range_function(q); }, 0, 1);
        splinedata[1] =
            splined_srf.generate([pot = pot](double q) { return pot->short_range_function_derivative(q); }, 0, 1);
        splinedata[2] = splined_srf.generate(
            [pot = pot](double q) { return pot->short_range_function_second_derivative(q); }, 0, 1);
        splinedata[3] =
            splined_srf.generate([pot = pot](double q) { return pot->short_range_function_third_derivative(q); }, 0, 1);
    }

  public:
    inline Splined() : EnergyImplementation<Splined>(Scheme::spline, infinity) {}

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
    void to_json(nlohmann::json &j) const { pot->to_json(j); }

  private:
    void _to_json(nlohmann::json &) const override {}
#endif
};

} // namespace CoulombGalore
