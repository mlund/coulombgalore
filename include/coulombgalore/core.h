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
#include <Eigen/Dense>

/** modern json for c++ added "_" suffix at around ~version 3.6 */
#ifdef NLOHMANN_JSON_HPP_
#define NLOHMANN_JSON_HPP
#endif

/** Namespace containing all of CoulombGalore */
namespace CoulombGalore {

/** Typedef for 3D vector such a position or dipole moment */
using vec3 = Eigen::Vector3d;
using mat33 = Eigen::Matrix3d;
using Tcomplex = Eigen::VectorXcd::value_type;

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

/**
 * @brief Returns the binomial coefficient
 * @f[
 *     {n \choose k}
 * @f]
 */
constexpr unsigned int binomial(signed int n, signed int k) { return factorial(n) / (factorial(k) * factorial(n - k)); }


/**
 * @brief Base class for truncation schemes
 *
 * This is the public interface used for dynamic storage of
 * different schemes. Energy functions are provided as virtual
 * functions which carries runtime overhead. The best performance
 * call these functions directly from the derived class.
 *
 * @fix Should give warning if 'has_dipolar_selfenergy' is false
 * @warning Charge neutralization scheme is not always implemented (or tested) for Yukawa-type potentials.
 */
class SchemeBase {
  private:
    std::array<double, 2> self_energy_prefactor; // Prefactor for self-energies, UNIT: [ 1 ]
    std::array<double, 2> self_field_prefactor;  // Prefactor for self-fields, UNIT: [ 1 ]

  protected:
    double inverse_cutoff = 0.0;       // inverse cutoff distance, UNIT: [ ( input length )^-1 ]
    double cutoff_squared = 0.0;       // square cutoff distance, UNIT: [ ( input length )^2 ]
    double inverse_debye_length = 0.0; // inverse Debye-length, UNIT: [ ( input length )^-1 ]
    double T0 = 0; // Spatial Fourier transformed modified interaction tensor, used to calculate the dielectric
    // constant, UNIT: [ 1 ]
    double chi = 0; // Negative integrated volume potential to neutralize charged system, UNIT: [ ( input length )^2 ]
    bool has_dipolar_selfenergy = false; // is there a valid dipolar self-energy?

    void setSelfEnergyPrefactor(const std::array<double, 2> &factor) {
        self_energy_prefactor = factor;
        selfEnergyFunctor = [invcutoff = inverse_cutoff,
                             factor = factor](const std::array<double, 2> &squared_moments) {
            double self_energy = 0.0;
            for (int i = 0; i < (int)squared_moments.size(); i++) {
                self_energy += factor[i] * squared_moments[i] * powi(invcutoff, 2 * i + 1);
            }
            return self_energy;
        };
    }

    void setSelfFieldPrefactor(const std::array<double, 2> &factor) {
        self_field_prefactor = factor;
        selfFieldFunctor = [invcutoff = inverse_cutoff, factor = factor](const std::array<vec3, 2> &moments) {
            vec3 self_field = {0.0, 0.0, 0.0};
            for (int i = 0; i < (int)moments.size(); i++) {
                self_field += factor[i] * moments[i] * powi(invcutoff, 2 * i + 1);
            }
            return self_field;
        };
    }

  public:
    std::string doi;     //!< DOI for original citation
    std::string name;    //!< Descriptive name
    Scheme scheme;       //!< Truncation scheme
    double cutoff;       //!< Cut-off distance, UNIT: [ input length ]
    double debye_length; //!< Debye-length, UNIT: [ input length ]

    std::function<double(const std::array<double, 2> &)> selfEnergyFunctor = nullptr; //!< Functor to calc. self-energy
    std::function<vec3(const std::array<vec3, 2> &)> selfFieldFunctor = nullptr;      //!< Functor to calc. self-field

    inline SchemeBase(Scheme scheme, double cutoff, double debye_length = infinity)
        : inverse_cutoff(1.0 / cutoff), cutoff_squared(cutoff * cutoff), inverse_debye_length(1.0 / debye_length),
          scheme(scheme), cutoff(cutoff), debye_length(debye_length) {}

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
    virtual double multipole_multipole_energy(double, double, const vec3 &, const vec3 &, const mat33 &, const mat33 &,
                                              const vec3 &) const = 0;

    virtual vec3 ion_field(double, const vec3 &) const = 0;
    virtual vec3 dipole_field(const vec3 &, const vec3 &) const = 0;
    virtual vec3 quadrupole_field(const mat33 &, const vec3 &) const = 0;
    virtual vec3 multipole_field(double, const vec3 &, const mat33 &, const vec3 &) const = 0;

    virtual vec3 ion_ion_force(double, double, const vec3 &) const = 0;
    virtual vec3 ion_dipole_force(double, const vec3 &, const vec3 &) const = 0;
    virtual vec3 dipole_dipole_force(const vec3 &, const vec3 &, const vec3 &) const = 0;
    virtual vec3 ion_quadrupole_force(double, const mat33 &, const vec3 &) const = 0;
    virtual vec3 multipole_multipole_force(double, double, const vec3 &, const vec3 &, const mat33 &, const mat33 &,
                                           const vec3 &) const = 0;

    // add remaining funtions here...

    // virtual double reciprocal_energy(const std::vector<vec3> &, const std::vector<double> &, const std::vector<vec3>
    // &, const vec3 &, int) = 0; virtual double surface_energy(const std::vector<vec3> &, const std::vector<double> &,
    // const std::vector<vec3> &, double) const = 0;

    // virtual vec3 reciprocal_force(const std::vector<vec3> &, const std::vector<double> &, const std::vector<vec3> &,
    // const vec3 &, int, const vec3 &, double, vec3 &) = 0; virtual vec3 surface_force(const std::vector<vec3> &, const
    // std::vector<double> &, const std::vector<vec3> &, double, double) const = 0;

    // virtual vec3 reciprocal_field(const std::vector<vec3> &, const std::vector<double> &, const std::vector<vec3> &,
    // const vec3 &, int, const vec3 &) = 0; virtual vec3 surface_field(const std::vector<vec3> &, const
    // std::vector<double> &, const std::vector<vec3> &, double) const = 0;

#ifdef NLOHMANN_JSON_HPP
  private:
    virtual void _to_json(nlohmann::json &) const = 0;

  public:
    inline void to_json(nlohmann::json &j) const {
        _to_json(j);
        if (std::isfinite(cutoff)) {
            j["cutoff"] = cutoff;
        }
        if (not doi.empty()) {
            j["doi"] = doi;
        }
        if (not name.empty()) {
            j["type"] = name;
        }
        if (std::isfinite(debye_length)) {
            j["debyelength"] = debye_length;
        }
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
        : SchemeBase(type, cutoff, debyelength) {}

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
            double q = r * inverse_cutoff;
            if (debyehuckel) // determined at compile time
            {
                return z / r * static_cast<const T *>(this)->short_range_function(q) *
                       std::exp(-inverse_debye_length * r);
            } else
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
        if (r2 < cutoff_squared) {
            const auto r1 = std::sqrt(r2);
            const auto q = r1 * inverse_cutoff;
            const auto kr = inverse_debye_length * r1;
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
        const auto r2 = r.squaredNorm();
        if (r2 < cutoff_squared) {
            const auto r1 = std::sqrt(r2);
            const auto q = r1 * inverse_cutoff;
            const auto q2 = q * q;
            const auto kr = inverse_debye_length * r1;
            const auto kr2 = kr * kr;
            const auto srf = static_cast<const T *>(this)->short_range_function(q);
            const auto dsrf = static_cast<const T *>(this)->short_range_function_derivative(q);
            const auto ddsrf = static_cast<const T *>(this)->short_range_function_second_derivative(q);

            const auto a = (srf * (1.0 + kr + kr2 / 3.0) - q * dsrf * (1.0 + 2.0 / 3.0 * kr) + q2 / 3.0 * ddsrf);
            const auto b = (srf * kr2 - 2.0 * kr * q * dsrf + ddsrf * q2) / 3.0;
            return 0.5 * ((3.0 / r2 * r.transpose() * quad * r - quad.trace()) * a + quad.trace() * b) / r2 / r1 *
                   std::exp(-inverse_debye_length * r1);
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
        const auto r2 = r.squaredNorm();
        if (r2 < cutoff_squared) {
            const auto r1 = std::sqrt(r2);
            const auto q = r1 * inverse_cutoff;
            const auto kr = inverse_debye_length * r1;
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
        const auto r2 = r.squaredNorm();
        if (r2 < cutoff_squared) {
            const auto r1 = std::sqrt(r2);
            const auto r3 = r1 * r2;
            const auto q = r1 * inverse_cutoff;
            const auto q2 = q * q;
            const auto kr = inverse_debye_length * r1;
            const auto kr2 = kr * kr;
            const auto srf = static_cast<const T *>(this)->short_range_function(q);
            const auto dsrf = static_cast<const T *>(this)->short_range_function_derivative(q);
            const auto ddsrf = static_cast<const T *>(this)->short_range_function_second_derivative(q);
            vec3 fieldD = (3.0 * mu.dot(r) * r / r2 - mu) / r3;
            fieldD *= (srf * (1.0 + kr + kr2 / 3.0) - q * dsrf * (1.0 + 2.0 / 3.0 * kr) + q2 / 3.0 * ddsrf);
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
    inline vec3 quadrupole_field(const mat33 &quad, const vec3 &r) const override {
        const auto r2 = r.squaredNorm();
        if (r2 < cutoff_squared) {
            const auto r_norm = std::sqrt(r2);
            vec3 r_hat = r / r_norm;
            const auto q = r_norm * inverse_cutoff;
            const auto q2 = q * q;
            const auto kr = inverse_debye_length * r_norm;
            const auto kr2 = kr * kr;
            const auto r4 = r2 * r2;
            vec3 quadrh = quad * r_hat;
            vec3 quadTrh = quad.transpose() * r_hat;

            double quadfactor = 1.0 / r2 * r.transpose() * quad * r;
            vec3 fieldD = 3.0 * ((5.0 * quadfactor - quad.trace()) * r_hat - quadrh - quadTrh) / r4;
            const auto srf = static_cast<const T *>(this)->short_range_function(q);
            const auto dsrf = static_cast<const T *>(this)->short_range_function_derivative(q);
            const auto ddsrf = static_cast<const T *>(this)->short_range_function_second_derivative(q);
            const auto dddsrf = static_cast<const T *>(this)->short_range_function_third_derivative(q);
            fieldD *= (srf * (1.0 + kr + kr2 / 3.0) - q * dsrf * (1.0 + 2.0 / 3.0 * kr) + q2 / 3.0 * ddsrf);
            vec3 fieldI = quadfactor * r_hat / r4;
            fieldI *= (srf * (1.0 + kr) * kr2 - q * dsrf * (3.0 * kr + 2.0) * kr + ddsrf * (1.0 + 3.0 * kr) * q2 -
                       q2 * q * dddsrf);
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
     *     {\bf E}(z,\boldsymbol{\mu}, {\bf r}) = \frac{z \hat{{\bf r}} }{|{\bf r}|^2} \left( s(q) - qs^{\prime}(q)
     * \right) + \frac{3 ( \boldsymbol{\mu} \cdot \hat{{\bf r}} ) \hat{{\bf r}} - \boldsymbol{\mu} }{|{\bf r}|^3} \left(
     * s(q) - qs^{\prime}(q)  + \frac{q^2}{3}s^{\prime\prime}(q) \right) + \frac{\boldsymbol{\mu}}{|{\bf
     * r}|^3}\frac{q^2}{3}s^{\prime\prime}(q)
     * @f]
     */
    inline vec3 multipole_field(double z, const vec3 &mu, const mat33 &quad, const vec3 &r) const override {
        const auto r2 = r.squaredNorm();
        if (r2 < cutoff_squared) {
            const auto r1 = std::sqrt(r2);
            vec3 rh = r / r1;
            const auto q = r1 * inverse_cutoff;
            const auto q2 = q * q;
            const auto r3 = r1 * r2;
            const auto kr = inverse_debye_length * r1;
            const auto kr2 = kr * kr;
            const double quadfactor = 1.0 / r2 * r.transpose() * quad * r;
            const auto srf = static_cast<const T *>(this)->short_range_function(q);
            const auto dsrf = static_cast<const T *>(this)->short_range_function_derivative(q);
            const auto ddsrf = static_cast<const T *>(this)->short_range_function_second_derivative(q);
            const auto dddsrf = static_cast<const T *>(this)->short_range_function_third_derivative(q);
            vec3 fieldIon = z * r / r3 * (srf * (1.0 + kr) - q * dsrf); // field from ion
            const auto postfactor =
                (srf * (1.0 + kr + kr2 / 3.0) - q * dsrf * (1.0 + 2.0 / 3.0 * kr) + q2 / 3.0 * ddsrf);
            vec3 fieldDd = (3.0 * mu.dot(r) * r / r2 - mu) / r3 * postfactor;
            vec3 fieldId = mu / r3 * (srf * kr2 - 2.0 * kr * q * dsrf + ddsrf * q2) / 3.0;
            vec3 fieldDq = 3.0 * ((5.0 * quadfactor - quad.trace()) * rh - quad * rh - quad.transpose() * rh) / r3 /
                           r1 * postfactor;
            vec3 fieldIq = quadfactor * rh / r3 / r1;
            fieldIq *= (srf * (1.0 + kr) * kr2 - q * dsrf * (3.0 * kr + 2.0) * kr + ddsrf * (1.0 + 3.0 * kr) * q2 -
                        q2 * q * dddsrf);
            return (fieldIon + fieldDd + fieldId + 0.5 * (fieldDq + fieldIq)) * std::exp(-kr);
        } else {
            return {0.0, 0.0, 0.0};
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
     * @param charge point charge, UNIT: [ input charge ]
     * @param dipole_moment dipole moment, UNIT: [ ( input length ) x ( input charge ) ]
     * @param r distance-vector between dipole and charge, @f$ {\bf r} = {\bf r}_{\mu} - {\bf r}_z @f$, UNIT: [ input
     * length ]
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
     * where @f$ {\bf E}(charge, {\bf r}) @f$ is the field from the ion at the location of the dipole.
     */
    inline double ion_dipole_energy(double charge, const vec3 &dipole_moment, const vec3 &r) const override {
        // Both expressions below gives same answer. Keep for possible optimization in future.
        // return -dipole_moment.dot(ion_field(charge,r)); // field from charge interacting with dipole
        return charge * dipole_potential(dipole_moment, -r); // potential of dipole interacting with charge
    }

    /**
     * @brief interaction energy between two point dipoles
     * @param dipole_moment_a dipole moment of particle A, UNIT: [ ( input length ) x ( input charge ) ]
     * @param dipole_moment_b dipole moment of particle B, UNIT: [ ( input length ) x ( input charge ) ]
     * @param r distance-vector between dipoles, @f$ {\bf r} = {\bf r}_{\mu_B} - {\bf r}_{\mu_A} @f$, UNIT: [ input
     * length ]
     * @returns interaction energy, UNIT: [ ( input charge )^2 / ( input length ) ]
     *
     * The interaction energy between two dipoles is decribed by
     * @f[
     *     u(\boldsymbol{\mu}_A, \boldsymbol{\mu}_B, {\bf r}) = -\boldsymbol{\mu}_A\cdot {\bf E}(\boldsymbol{\mu}_B,
     * {\bf r})
     * @f]
     * where @f$ {\bf E}(\boldsymbol{\mu}_B, {\bf r}) @f$ is the field from dipole B at the location of dipole A.
     */
    inline double dipole_dipole_energy(const vec3 &dipole_moment_a, const vec3 &dipole_moment_b,
                                       const vec3 &r) const override {
        return -dipole_moment_a.dot(dipole_field(dipole_moment_b, r));
    }

    /**
     * @brief interaction energy between a point charges and a point quadrupole
     * @param charge point charge, UNIT: [ input charge ]
     * @param quad quadrupole moment, UNIT: [ ( input length )^2 x ( input charge ) ]
     * @param r distance-vector between quadrupole and charge, @f$ {\bf r} = {\bf r}_{\boldsymbol{Q}} - {\bf r}_z @f$,
     * UNIT: [ input length ]
     * @returns interaction energy, UNIT: [ ( input charge )^2 / ( input length ) ]
     *
     * The interaction energy between an ion and a quadrupole is decribed by
     * @f[
     *     u(z, \boldsymbol{Q}, {\bf r}) = z \Phi(\boldsymbol{Q}, -{\bf r})
     * @f]
     * where @f$ \Phi(\boldsymbol{Q}, -{\bf r}) @f$ is the potential from the quadrupole at the location of the ion.
     */
    inline double ion_quadrupole_energy(double charge, const mat33 &quad, const vec3 &r) const override {
        return charge * quadrupole_potential(quad, -r); // potential of quadrupole interacting with charge
    }

    /**
     * @brief interaction energy between two multipoles with charges and dipole moments
     * @param zA point charge of particle A, UNIT: [ input charge ]
     * @param zB point charge of particle B, UNIT: [ input charge ]
     * @param muA point dipole moment of particle A, UNIT: [ ( input length ) x ( input charge ) ]
     * @param muB point dipole moment of particle B, UNIT: [ ( input length ) x ( input charge ) ]
     * @param quadA point quadrupole of particle A, UNIT: [ ( input length )^2 x ( input charge ) ]
     * @param quadB point quadrupole of particle B, UNIT: [ ( input length )^2 x ( input charge ) ]
     * @param r distance-vector between dipoles, @f$ {\bf r} = {\bf r}_{\mu_B} - {\bf r}_{\mu_A} @f$, UNIT: [ input
     * length ]
     * @returns interaction energy, UNIT: [ ( input charge )^2 / ( input length ) ]
     *
     * A combination of the functions 'ion_ion_energy', 'ion_dipole_energy', 'dipole_dipole_energy' and
     * 'ion_quadrupole_energy'.
     */
    inline double multipole_multipole_energy(double zA, double zB, const vec3 &muA, const vec3 &muB, const mat33 &quadA,
                                             const mat33 &quadB, const vec3 &r) const override {
        const double r2 = r.squaredNorm();
        if (r2 < cutoff_squared) {
            const double r1 = std::sqrt(r2);
            const double q = r1 / cutoff;
            const double kr = inverse_debye_length * r1;
            const double quadAtrace = quadA.trace();
            const double quadBtrace = quadB.trace();

            const double srf = static_cast<const T *>(this)->short_range_function(q);
            const double dsrfq = static_cast<const T *>(this)->short_range_function_derivative(q) * q;
            const double ddsrfq2 =
                static_cast<const T *>(this)->short_range_function_second_derivative(q) * q * q / 3.0;

            const double angcor = (srf * (1.0 + kr) - dsrfq);
            const double unicor = (srf * kr * kr / 3.0 - 2.0 / 3.0 * dsrfq * kr + ddsrfq2);
            const double muBdotr = muB.dot(r);
            vec3 field_dipoleB = (3.0 * muBdotr * r / r2 - muB) * (angcor + unicor) + muB * unicor;

            const double ion_ion = zA * zB * srf * r2;                           // will later be divided by r3
            const double ion_dipole = (zB * muA.dot(r) - zA * muBdotr) * angcor; // will later be divided by r3
            const double dipole_dipole = -muA.dot(field_dipoleB);                // will later be divided by r3
            double ion_quadrupole = zA * 0.5 *
                                    ((3.0 / r2 * r.transpose() * quadB * r - quadBtrace) * (angcor + unicor) +
                                     quadBtrace * unicor); // will later be divided by r3
            ion_quadrupole +=
                zB * 0.5 *
                ((3.0 / r2 * r.transpose() * quadA * r - quadAtrace) * (angcor + unicor) + quadAtrace * unicor);

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
     * @param charge charge, UNIT: [ input charge ]
     * @param mu dipole moment, UNIT: [ ( input length ) x ( input charge ) ]
     * @param r distance-vector between dipole and charge, @f$ {\bf r} = {\bf r}_{\mu} - {\bf r}_z @f$, UNIT: [ input
     * length ]
     * @returns interaction force, UNIT: [ ( input charge )^2 / ( input length )^2 ]
     *
     * @details The force between an ion and a dipole is decribed by
     * @f[
     *     {\bf F}(z, \boldsymbol{\mu}, {\bf r}) = z {\bf E}(\boldsymbol{\mu}, {\bf r})
     * @f]
     * where @f$ {\bf E}(\boldsymbol{\mu}, {\bf r}) @f$ is the field from the dipole at the location of the ion.
     */
    inline vec3 ion_dipole_force(double charge, const vec3 &mu, const vec3 &r) const override {
        return charge * dipole_field(mu, r);
    }

    /**
     * @brief interaction force between two point dipoles
     * @param muA dipole moment of particle A, UNIT: [ ( input length ) x ( input charge ) ]
     * @param muB dipole moment of particle B, UNIT: [ ( input length ) x ( input charge ) ]
     * @param r distance-vector between dipoles, @f[ {\bf r} = {\bf r}_{\mu_B} - {\bf r}_{\mu_A} @f], UNIT: [ input
     * length ]
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
        const double r2 = r.squaredNorm();
        if (r2 < cutoff_squared) {
            const double r1 = std::sqrt(r2);
            vec3 rh = r / r1;
            const double q = r1 * inverse_cutoff;
            const double q2 = q * q;
            const double kr = inverse_debye_length * r1;
            const double r4 = r2 * r2;
            const double muAdotRh = muA.dot(rh);
            const double muBdotRh = muB.dot(rh);
            vec3 forceD =
                3.0 * ((5.0 * muAdotRh * muBdotRh - muA.dot(muB)) * rh - muBdotRh * muA - muAdotRh * muB) / r4;
            const double srf = static_cast<const T *>(this)->short_range_function(q);
            const double dsrf = static_cast<const T *>(this)->short_range_function_derivative(q);
            const double ddsrf = static_cast<const T *>(this)->short_range_function_second_derivative(q);
            const double dddsrf = static_cast<const T *>(this)->short_range_function_third_derivative(q);
            forceD *= (srf * (1.0 + kr + kr * kr / 3.0) - q * dsrf * (1.0 + 2.0 / 3.0 * kr) + q2 / 3.0 * ddsrf);
            vec3 forceI = muAdotRh * muBdotRh * rh / r4;
            forceI *= (srf * (1.0 + kr) * kr * kr - q * dsrf * (3.0 * kr + 2.0) * kr + ddsrf * (1.0 + 3.0 * kr) * q2 -
                       q2 * q * dddsrf);
            return (forceD + forceI) * std::exp(-kr);
        } else {
            return {0.0, 0.0, 0.0};
        }
    }

    /**
     * @brief interaction force between a point charge and a point quadrupole
     * @param charge point charge, UNIT: [ input charge ]
     * @param quad point quadrupole, UNIT: [ ( input length )^2 x ( input charge ) ]
     * @param r distance-vector between particles, @f$ {\bf r} = {\bf r}_{Q} - {\bf r}_{charge} @f$, UNIT: [ input
     * length ]
     * @returns interaction force, UNIT: [ ( input charge )^2 / ( input length )^2 ]
     *
     * The force between a point charge and a point quadrupole is described by
     * @f[
     *     {\bf F}(z, Q, {\bf r}) = z {\bf E}(Q, {\bf r})
     * @f]
     * where @f$ {\bf E}(Q, {\bf r}) @f$ is the field from the quadrupole at the location of the ion.
     */
    inline vec3 ion_quadrupole_force(double charge, const mat33 &quad, const vec3 &r) const override {
        return charge * quadrupole_field(quad, r);
    }

    /**
     * @brief interaction force between two point multipoles
     * @param zA charge of particle A, UNIT: [ input charge ]
     * @param zB charge of particle B, UNIT: [ input charge ]
     * @param muA dipole moment of particle A, UNIT: [ ( input length ) x ( input charge ) ]
     * @param muB dipole moment of particle B, UNIT: [ ( input length ) x ( input charge ) ]
     * @param quadA point quadrupole of particle A, UNIT: [ ( input length )^2 x ( input charge ) ]
     * @param quadB point quadrupole of particle B, UNIT: [ ( input length )^2 x ( input charge ) ]
     * @param r distance-vector between dipoles, @f[ {\bf r} = {\bf r}_{\mu_B} - {\bf r}_{\mu_A} @f], UNIT: [ input
     * length ]
     * @returns interaction force, UNIT: [ ( input charge )^2 / ( input length )^2 ]
     *
     * @details A combination of the functions 'ion_ion_force', 'ion_dipole_force', 'dipole_dipole_force' and
     * 'ion_quadrupole_force'.
     */
    inline vec3 multipole_multipole_force(double zA, double zB, const vec3 &muA, const vec3 &muB, const mat33 &quadA,
                                          const mat33 &quadB, const vec3 &r) const override {
        const double r2 = r.squaredNorm();
        if (r2 < cutoff_squared) {
            const double r1 = std::sqrt(r2);
            const double q = r1 * inverse_cutoff;
            const double q2 = q * q;
            const double kr = inverse_debye_length * r1;
            vec3 rh = r / r1;
            const double muAdotRh = muA.dot(rh);
            const double muBdotRh = muB.dot(rh);

            const double srf = static_cast<const T *>(this)->short_range_function(q);
            const double dsrfq = static_cast<const T *>(this)->short_range_function_derivative(q) * q;
            const double ddsrfq2 = static_cast<const T *>(this)->short_range_function_second_derivative(q) * q2 / 3.0;
            const double dddsrfq3 = static_cast<const T *>(this)->short_range_function_third_derivative(q) * q2 * q;

            const double angcor = (srf * (1.0 + kr) - dsrfq);
            const double unicor = (srf * kr - 2.0 * dsrfq) * kr / 3.0 + ddsrfq2;
            const double totcor = unicor + angcor;
            const double r3corr =
                (angcor * kr * kr - dsrfq * 2.0 * (1.0 + kr) * kr + 3.0 * ddsrfq2 * (1.0 + 3.0 * kr) - dddsrfq3);

            vec3 ion_ion = zB * zA * r * angcor * r1;
            vec3 ion_dipole = zA * ((3.0 * muBdotRh * rh - muB) * totcor + muB * unicor);
            ion_dipole += zB * ((3.0 * muAdotRh * rh - muA) * totcor + muA * unicor);
            ion_dipole *= r1;
            vec3 forceD =
                3.0 * ((5.0 * muAdotRh * muBdotRh - muA.dot(muB)) * rh - muBdotRh * muA - muAdotRh * muB) * totcor;
            vec3 dipole_dipole = (forceD + muAdotRh * muBdotRh * rh * r3corr);
            double quadfactor = 1.0 / r2 * r.transpose() * quadB * r;
            vec3 fieldD =
                3.0 * (-(5.0 * quadfactor - quadB.trace()) * rh + quadB * rh + quadB.transpose() * rh) * totcor;
            vec3 ion_quadrupole = zA * 0.5 * (fieldD - quadfactor * rh * r3corr);
            quadfactor = 1.0 / r2 * r.transpose() * quadA * r;
            fieldD = 3.0 * ((5.0 * quadfactor - quadA.trace()) * rh - quadA * rh - quadA.transpose() * rh) * totcor;
            ion_quadrupole += zB * 0.5 * (fieldD + quadfactor * rh * r3corr);

            return (ion_ion + ion_dipole + dipole_dipole + ion_quadrupole) * std::exp(-kr) / r2 / r2;
        } else {
            return {0.0, 0.0, 0.0};
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
     * @param squared_moments vector with square moments, i.e. charge squared and dipole moment squared, UNIT: [ ( input
     * charge )^2 , ( input length )^2 x ( input charge )^2 , ... ]
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
     * @brief Self-field for all type of interactions
     * @param moments vector with moments, i.e. charge and dipole moment, UNIT: [ input charge , input length x input
     * charge , ... ]
     * @returns self-field, UNIT: [ input charge / ( input length )^2 ]
     *
     * @details The self-field is described by
     * @f$
     *     {\bf E}(self) = ...
     * @f$
     * where @f$ p_i @f$ is the prefactor for the self-energy for species 'i'.
     * Here i=0 represent ions, i=1 represent dipoles etc.
     */
    inline vec3 self_field(const std::array<vec3, 2> &moments) const {
        assert(selfFieldFunctor != nullptr);
        return selfFieldFunctor(moments);
    }

    /**
     * @brief Compensating term for non-neutral systems
     * @param charges Charges of particles, UNIT: [ input charge ]
     * @param volume Volume of unit-cell, UNIT: [ ( input length )^3 ]
     * @returns energy, UNIT: [ ( input charge )^2 / ( input length ) ]
     * @note DOI:10.1021/jp951011v
     */
    inline double neutralization_energy(const std::vector<double> &charges, double volume) const override {
        const auto charge_sum = std::accumulate(charges.begin(), charges.end(), 0.0);
        return ((this)->chi / 2.0 / volume * charge_sum * charge_sum);
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
        has_dipolar_selfenergy = true;
        doi = "10.1063/1.478738";
        alphaRed = alpha * cutoff;
        alphaRed2 = alphaRed * alphaRed;
        setSelfEnergyPrefactor(
            {-alphaRed / pi_sqrt - std::erfc(alphaRed) / 2.0, -powi(alphaRed, 3) * 2.0 / 3.0 / pi_sqrt});
        setSelfFieldPrefactor({0.0, 0.0});
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

} // namespace CoulombGalore
