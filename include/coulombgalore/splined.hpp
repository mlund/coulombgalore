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
    inline T eval(const typename base::data &d, const T r2) const {
        assert(r2 != 0); // r2 cannot be *exactly* zero
        const size_t pos = std::lower_bound(d.r2.begin(), d.r2.end(), r2) - d.r2.begin() - 1;
        const size_t pos6 = 6 * pos;
        assert((pos6 + 5) < d.c.size() && "out of bounds");
        const T dz = r2 - d.r2[pos];
        return d.c[pos6] +
               dz * (d.c[pos6 + 1] +
                     dz * (d.c[pos6 + 2] + dz * (d.c[pos6 + 3] + dz * (d.c[pos6 + 4] + dz * (d.c[pos6 + 5])))));
    }

    /*
     * @brief Get tabulated value at df(x)/dx
     * @param d Table data
     * @param r2 value
     */
    T evalDer(const typename base::data &d, const T r2) const {
        const size_t pos = std::lower_bound(d.r2.begin(), d.r2.end(), r2) - d.r2.begin() - 1;
        const size_t pos6 = 6 * pos;
        const T dz = r2 - d.r2[pos];
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
    inline Splined() : EnergyImplementation<Splined>(Scheme::spline, infinity) { setTolerance(1e-3); }

    /**
     * @brief Returns vector with number of spline knots the short-range-function and its derivatives
     */
    inline std::vector<size_t> numKnots() const {
        std::vector<size_t> n;
        for (auto &i : splinedata) {
            n.push_back(i.numKnots());
        }
        return n;
    }

    /**
     * @brief Set relative spline tolerance
     */
    inline void setTolerance(double tol) { splined_srf.setTolerance(tol); }

    /**
     * @brief Spline given potential type
     * @tparam T Potential class
     * @param args Passed to constructor of potential class
     * @note This must be called before using any other functions
     */
    template <class T, class... Args> void spline(Args &&...args) {
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
