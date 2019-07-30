#pragma once

#include <string>
#include <limits>
#include <cmath>
#include <iostream>
#include <Eigen/Core>

namespace CoulombGalore {

typedef Eigen::Vector3d Point; //!< typedef for 3d vector

constexpr double infty = std::numeric_limits<double>::infinity(); //!< Numerical infinity

/**
 * @brief Returns the factorial of 'n'. Note that 'n' must be positive semidefinite.
 * @note Calculated at compile time and thus have no run-time overhead.
 */
constexpr unsigned int factorial(unsigned int n) { return n <= 1 ? 1 : n * factorial(n - 1); }
#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("[Faunus] Factorial") {
    CHECK(factorial(0) == 1);
    CHECK(factorial(1) == 1);
    CHECK(factorial(2) == 2);
    CHECK(factorial(3) == 6);
    CHECK(factorial(10) == 3628800);
}
#endif

constexpr unsigned int binomial(unsigned int n, unsigned int k) { return factorial( n ) / factorial( k ) / factorial( n - k ); }
#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("[Faunus] Binomial") {
    CHECK(binomial(3,2) == 3);
    CHECK(binomial(5,2) == 10);
    CHECK(binomial(8,3) == 56);
    CHECK(binomial(9,7) == 36);
    CHECK(binomial(5,0) == 1);
    CHECK(binomial(12,1) == 12);
    CHECK(binomial(11,11) == 1);
    CHECK(binomial(2,0) == 1);
    CHECK(binomial(3,1) == 3);
    CHECK(binomial(4,2) == 6);
    CHECK(binomial(5,3) == 10);
}
#endif

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
    for(int n = 1; n < P + 1; n++) {
        double val = 0.0;
        for(int k = 1; k < n + l + 1; k++)
            val += std::pow( q , k - 1 );
        Ct *= val;
    }
    double Dt = std::pow( 1.0 - q , P ); // (1-q)^P
    return ( Ct * Dt );
}

/**
 * @brief Gives the derivative of the q-Pochhammer Symbol
 */
inline double qPochhammerSymbolDerivative(double q, int l = 0, int P = 300) {
    double Ct = 1.0; // evaluates to \prod_{n=1}^P\sum_{k=0}^{n+l}q^k
    for(int n = 1; n < P + 1; n++) {
        double val = 0.0;
        for(int k = 1; k < n + l + 1; k++)
            val += std::pow( q , k - 1 );
        Ct *= val;
    }
    double dCt = 0.0; // evaluates to derivative of \prod_{n=1}^P\sum_{k=0}^{n+l}q^k
    for(int n = 1; n < P + 1; n++) {
        double nom = 0.0;
        double denom = 1.0;
        for(int k = 2; k < n + l + 1; k++) {
            nom += ( k - 1 ) * std::pow( q , k - 2 );
            denom += std::pow( q , k - 1 );
        }
        dCt += nom / denom;
    }
    dCt *= Ct;
    double Dt = std::pow( 1.0 - q , P ); // (1-q)^P
    double dDt = 0.0;
    if(P > 0)
        dDt = -P * std::pow( 1 - q , P - 1 ); // derivative of (1-q)^P
    return ( Ct * dDt + dCt * Dt );
}

/**
 * @brief Gives the second derivative of the q-Pochhammer Symbol
 */
inline double qPochhammerSymbolSecondDerivative(double q, int l = 0, int P = 300) {
    double Ct = 1.0; // evaluates to \prod_{n=1}^P\sum_{k=0}^{n+l}q^k
    double DS = 0.0;
    double dDS = 0.0;
    for(int n = 1; n < P + 1; n++) {
        double tmp = 0.0;
        for(int k = 1; k < n + l + 1; k++)
            tmp += std::pow( q , k - 1 );
        Ct *= tmp;
        double nom = 0.0;
        double denom = 1.0;
        for(int k = 2; k < n + l + 1; k++) {
            nom += ( k - 1 ) * std::pow( q , k - 2 );
            denom += std::pow( q , k - 1 );
        }
        DS += nom / denom;
        double diffNom = 0.0;
        double diffDenom = 1.0;
        for(int k = 3; k < n + l + 1; k++) {
            diffNom += ( k - 1 ) * ( k - 2 ) * std::pow( q , k - 3 );
            diffDenom += ( k - 1) * std::pow( q , k - 2 );
        }
        dDS += ( diffNom * denom - nom * diffDenom ) / denom / denom;
    }
    double dCt = Ct * DS; // derivative of \prod_{n=1}^P\sum_{k=0}^{n+l}q^k
    double ddCt = dCt * DS + Ct * dDS; // second derivative of \prod_{n=1}^P\sum_{k=0}^{n+l}q^k
    double Dt = std::pow( 1.0 - q , P ); // (1-q)^P
    double dDt = 0.0;
    if(P > 0)
        dDt = -P * std::pow( 1 - q , P - 1 ); // derivative of (1-q)^P
    double ddDt = 0.0;
    if(P > 1)
        ddDt = P * ( P - 1 ) * std::pow( 1 - q , P - 2 ); // second derivative of (1-q)^P
    return (Ct*ddDt + 2*dCt*dDt + ddCt*Dt);
}

/**
 * @brief Gives the third derivative of the q-Pochhammer Symbol
 */
inline double qPochhammerSymbolThirdDerivative(double q, int l = 0, int P = 300) {
    double Ct = 1.0; // evaluates to \prod_{n=1}^P\sum_{k=0}^{n+l}q^k
    double DS = 0.0;
    double dDS = 0.0;
    double ddDS = 0.0;
    for(int n = 1; n < P + 1; n++) {
        double tmp = 0.0;
        for(int k = 1; k < n + l + 1; k++)
            tmp += std::pow( q , k - 1 );
        Ct *= tmp;
        double f = 0.0;
        double g = 1.0;
        for(int k = 2; k < n + l + 1; k++) {
            f += ( k - 1 ) * std::pow( q , k - 2 );
            g += std::pow( q , k - 1 );
        }
        DS += f / g;
        double df = 0.0;
        double dg = 0.0;
        if( n + l > 1)
            dg = 1.0;
        for(int k = 3; k < n + l + 1; k++) {
            df += ( k - 1 ) * ( k - 2 ) * std::pow( q , k - 3 );
            dg += ( k - 1) * std::pow( q , k - 2 );
        }
        dDS += ( df * g - f * dg ) / g / g;
        double ddf = 0.0;
        double ddg = 0.0;
        if( n + l > 2)
            ddg = 2.0;
        for(int k = 4; k < n + l + 1; k++) {
            ddf += ( k - 1 ) * ( k - 2 ) * ( k - 3 ) * std::pow( q , k - 4 );
            ddg += ( k - 1) * ( k - 2 ) * std::pow( q , k - 3 );
        }
        ddDS += ( ddf * g * g - 2.0 * df * dg * g + 2.0 * f * dg*dg - f * ddg * g ) / g / g / g;
    }
    double dCt = Ct * DS; // derivative of \prod_{n=1}^P\sum_{k=0}^{n+l}q^k
    double ddCt = dCt * DS + Ct * dDS; // second derivative of \prod_{n=1}^P\sum_{k=0}^{n+l}q^k
    double dddCt = ddCt * DS + 2.0*dCt*dDS + Ct * ddDS; // third derivative of \prod_{n=1}^P\sum_{k=0}^{n+l}q^k
    double Dt = std::pow( 1.0 - q , P ); // (1-q)^P
    double dDt = 0.0;
    if(P > 0)
        dDt = -P * std::pow( 1 - q , P - 1 ); // derivative of (1-q)^P
    double ddDt = 0.0;
    if(P > 1)
        ddDt = P * ( P - 1 ) * std::pow( 1 - q , P - 2 ); // second derivative of (1-q)^P
    double dddDt = 0.0;
    if(P > 2)
        dddDt = -P * ( P - 1 ) * ( P - 2 ) * std::pow( 1 - q , P - 3 ); // third derivative of (1-q)^P
    return ( dddCt*Dt + 3.0*ddCt*dDt + 3*dCt*ddDt + Ct*dddDt );
}

#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("qPochhammerSymbol") {
    using doctest::Approx;
    CHECK(qPochhammerSymbol(0.5, 0, 0) == 1);
    CHECK(qPochhammerSymbol(0, 0, 1) == 1);
    CHECK(qPochhammerSymbol(1, 0, 1) == 0);
    CHECK(qPochhammerSymbol(1, 1, 2) == 0);
    CHECK(qPochhammerSymbol(0.75, 0, 2) == Approx(0.109375));
    CHECK(qPochhammerSymbol(2.0/3.0, 2, 5) == Approx(0.4211104676));
    CHECK(qPochhammerSymbol(0.125, 1, 1) == Approx(0.984375));
    CHECK(qPochhammerSymbolDerivative(0.75, 0, 2) == Approx(-0.8125));
    CHECK(qPochhammerSymbolDerivative(2.0/3.0, 2, 5) == Approx(-2.538458169));
    CHECK(qPochhammerSymbolDerivative(0.125, 1, 1) == Approx(-0.25));
    CHECK(qPochhammerSymbolSecondDerivative(0.75, 0, 2) == Approx(2.5));
    CHECK(qPochhammerSymbolSecondDerivative(2.0/3.0, 2, 5) == Approx(-1.444601767));
    CHECK(qPochhammerSymbolSecondDerivative(0.125, 1, 1) == Approx(-2.0));
    CHECK(qPochhammerSymbolThirdDerivative(0.75, 0, 2) == Approx(6.0));
    CHECK(qPochhammerSymbolThirdDerivative(2.0/3.0, 2, 5) == Approx(92.48631425));
    CHECK(qPochhammerSymbolThirdDerivative(0.125, 1, 1) == Approx(0.0));
    CHECK(qPochhammerSymbolThirdDerivative(0.4, 3, 7) == Approx(-32.80472205));
}
#endif

/**
 * @brief Base class for truncation schemes
 *
 * Derived classes must implement the splitting function which
 * does not need to be highly optimized as it will later be splined.
 */
class SchemeBase {
  public:
    enum class TruncationScheme { plain, ewald, wolf, poisson, qpotential, fanourgakis };
    std::string doi;                  //!< DOI for original citation
    std::string name;                 //!< Descriptive name
    TruncationScheme scheme;          //!< Truncation scheme
    double cutoff;                    //!< Cut-off distance
    double debye_length;              //!< Debye-length
    double T0;                        //!< Spatial Fourier transformed modified interaction tensor, used to calculate the dielectric constant
    std::array<double,2> self_energy_prefactor; //!< Prefactor for self-energies
    inline SchemeBase(TruncationScheme scheme, double cutoff, double debye_length=infty) : scheme(scheme), cutoff(cutoff), debye_length(debye_length) {}

    /**
     * @brief Short-range function
     * @param q q = r / Rcutoff
     */
    virtual double short_range_function(double) const = 0;

    virtual double short_range_function_derivative(double q, double dh=1e-6) {
        return ( short_range_function(q + dh) - short_range_function(q - dh) ) / ( 2 * dh );
    }

    virtual double short_range_function_second_derivative(double q, double dh=1e-6) {
        return ( short_range_function_derivative(q + dh , dh ) - short_range_function_derivative(q - dh , dh ) ) / ( 2 * dh );
    }

    virtual double short_range_function_third_derivative(double q, double dh=1e-6) {
        return ( short_range_function_second_derivative(q + dh , dh ) - short_range_function_second_derivative(q - dh , dh ) ) / ( 2 * dh );
    }

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
     * @f[ k_B @f] the Boltzmann constant, and @f[ T @f] the temperature.
     */
    double calc_dielectric(double M2V) { return ( M2V * T0 + 2.0 * M2V + 1.0 ) / ( M2V * T0 - M2V + 1.0 ); }

#ifdef NLOHMANN_JSON_HPP
  private:
    virtual void _to_json(nlohmann::json &) const = 0;
    virtual void _from_json(const nlohmann::json &) = 0;

  public:
    inline void from_json(const nlohmann::json &j) {
        if (scheme != TruncationScheme::plain)
            cutoff = j.at("cutoff").get<double>();
        _from_json(j);
    }
    inline void to_json(nlohmann::json &j) const {
        _to_json(j);
        if (cutoff < infty)
            j["cutoff"] = cutoff;
        if (not doi.empty())
            j["doi"] = doi;
        if (not name.empty())
            j["type"] = name;
    }
#endif
};

/**
 * @brief Class for calculation of interaction energies
 * @todo Replace this with a splined version
 *
 * @details In the following @f[ s(q) @f] is a short-ranged function and @f[ q = r / R_c @f] where @f[ R_c @f] is the cut-off distance.
 */
template <class Tscheme> class PairPotential : public Tscheme {
  private:
    double invcutoff; // inverse cutoff distance
    double cutoff2;   // square cutoff distance
    double kappa;     // inverse Debye-length
  public:
    using Tscheme::cutoff;
    using Tscheme::debye_length;
    using Tscheme::self_energy_prefactor;
    using Tscheme::short_range_function;
    using Tscheme::short_range_function_derivative;
    using Tscheme::short_range_function_second_derivative;
    using Tscheme::short_range_function_third_derivative;

    template <class... Args> PairPotential(Args &&... args) : Tscheme(args...) {
      invcutoff = 1.0 / cutoff;
      cutoff2 = cutoff*cutoff;
      kappa = 1.0 / debye_length;
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
    inline double ion_potential(double z, double r) {
        if ( r < cutoff ) {
            double q = r * invcutoff;
            return z / r * short_range_function(q) * std::exp( -kappa * r);
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
     *     \Phi(\boldsymbol{\mu}, {\bf r}) = \frac{\boldsymbol{\mu} \cdot \hat{{\bf r}} }{|{\bf r}|^2} \left( s(q) - qs^{\prime}(q) \right)
     * @f]
     */
    inline double dipole_potential(Point mu, Point r) {
        double r2 = r.squaredNorm();
        if ( r2 < cutoff2 ) {
            double r1 = std::sqrt(r2);
            double q = r1 * invcutoff;
            return mu.dot(r) / r2 / r1  * ( short_range_function(q) * ( 1.0 + kappa * r1 ) - q * short_range_function_derivative(q) ) * std::exp( -kappa * r1);
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
    inline Point ion_field(double z, Point r) {
        double r2 = r.squaredNorm();
        if ( r2 < cutoff2 ) {
            double r1 = std::sqrt(r2);
            double q = r1 * invcutoff;
            return z * r / r2 / r1 * ( short_range_function(q) * ( 1.0 + kappa * r1 )  - q * short_range_function_derivative(q) ) * std::exp( -kappa * r1);
        } else {
            return {0,0,0};
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
     *     {\bf E}(\boldsymbol{\mu}, {\bf r}) = \frac{3 ( \boldsymbol{\mu} \cdot \hat{{\bf r}} ) \hat{{\bf r}} - \boldsymbol{\mu} }{|{\bf r}|^3} \left( s(q) - qs^{\prime}(q)  + \frac{q^2}{3}s^{\prime\prime}(q) \right) + \frac{\boldsymbol{\mu}}{|{\bf r}|^3}\frac{q^2}{3}s^{\prime\prime}(q)
     * @f]
     */
    inline Point dipole_field(Point mu, Point r) {
        double r2 = r.squaredNorm();
        if ( r2 < cutoff2 ) {
            double r1 = std::sqrt(r2);
            double q = r1 * invcutoff;
	    double srf = short_range_function(q);
	    double dsrf = short_range_function_derivative(q);
	    double ddsrf = short_range_function_second_derivative(q);
            Point field = ( 3.0 * mu.dot(r) * r / r2 - mu ) / r2 / r1;
	    field *= ( srf * ( 1.0 + kappa * r1 + kappa * kappa * r2 / 3.0 ) - q * dsrf * ( 1.0 + 2.0 / 3.0 * kappa * r1 ) + q * q / 3.0 * ddsrf );
            field += mu / r2 / r1 * ( srf * kappa * kappa * r2 - 2.0 * kappa * r1 * q * dsrf +  ddsrf * q * q ) / 3.0;
            return field * std::exp( -kappa * r1);
        } else {
            return {0,0,0};
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
    inline double ion_ion_energy(double zA, double zB, double r) {
        return zB * ion_potential(zA,r);
    }

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
    inline double ion_dipole_energy(double z, Point mu, Point r) {
        // Both expressions below gives same answer. Keep for possible optimization in future.
        //return -mu.dot(ion_field(z,r)); // field from charge interacting with dipole
        return z * dipole_potential(mu,-r); // potential of dipole interacting with charge
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
     *     u(\boldsymbol{\mu}_A, \boldsymbol{\mu}_B, {\bf r}) = -\boldsymbol{\mu}_A\cdot {\bf E}(\boldsymbol{\mu}_B, {\bf r})
     * @f]
     * where @f[ {\bf E}(\boldsymbol{\mu}_B, {\bf r}) @f] is the field from dipole B at the location of dipole A.
     */
    inline double dipole_dipole_energy(Point muA, Point muB, Point r) {
        return -muA.dot(dipole_field(muB,r));
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
    inline Point ion_ion_force(double zA, double zB, Point r) {
        return zB * ion_field(zA,r);
    }

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
    inline Point ion_dipole_force(double z, Point mu, Point r) {
        return z * dipole_field(mu,r);
    }

    /**
     * @brief dipole-dipole interaction energy
     * @returns interaction energy in electrostatic units ( why not Hartree atomic units? )
     * @param muA dipole moment of particle A
     * @param muB dipole moment of particle B
     * @param r distance-vector between dipoles, @f[ {\bf r} = {\bf r}_{\mu_B} - {\bf r}_{\mu_A} @f]
     *
     * @details The force between two dipoles is decribed by
     * @f[
     *     {\bf F}(\boldsymbol{\mu}_A, \boldsymbol{\mu}_B, {\bf r}) = {\bf F}_D(\boldsymbol{\mu}_A, \boldsymbol{\mu}_B, {\bf r})\left( s(q) - qs^{\prime}(q)  + \frac{q^2}{3}s^{\prime\prime}(q) \right) + {\bf F}_I(\boldsymbol{\mu}_A, \boldsymbol{\mu}_B, {\bf r})\left( s^{\prime\prime}(q)  - qs^{\prime\prime\prime}(q) \right)q^2
     * @f]
     * where the 'direct' (D) force contribution is
     * @f[
     *     {\bf F}_D(\boldsymbol{\mu}_A, \boldsymbol{\mu}_B, {\bf r}) = 3\frac{ 5 (\boldsymbol{\mu}_A \cdot {\bf \hat{r}}) (\boldsymbol{\mu}_B \cdot {\bf \hat{r}}){\bf \hat{r}} - (\boldsymbol{\mu}_A \cdot \boldsymbol{\mu}_B){\bf \hat{r}} - (\boldsymbol{\mu}_A \cdot {\bf \hat{r}})\boldsymbol{\mu}_B - (\boldsymbol{\mu}_B \cdot {\bf \hat{r}})\boldsymbol{\mu}_A }{|{\bf r}|^4}
     * @f]
     * and the 'indirect' (I) force contribution is
     * @f[
     *     {\bf F}_I(\boldsymbol{\mu}_A, \boldsymbol{\mu}_B, {\bf r}) = \frac{ (\boldsymbol{\mu}_A \cdot {\bf \hat{r}}) (\boldsymbol{\mu}_B \cdot {\bf \hat{r}}){\bf \hat{r}}}{|{\bf r}|^4}.
     * @f]
     */
    inline Point dipole_dipole_force(Point muA, Point muB, Point r) {
        double r2 = r.squaredNorm();
        if ( r2 < cutoff2 ) {
            double r1 = std::sqrt(r2);
            Point rh = r / r1;
            double q = r1 * invcutoff;
            double r4 = r2 * r2;
            double muAdotRh = muA.dot(rh);
            double muBdotRh = muB.dot(rh);
            Point forceD = 3.0 * ( ( 5.0 * muAdotRh*muBdotRh - muA.dot(muB) ) * rh - muBdotRh * muA - muAdotRh * muB  ) / r4;
	    double srf = short_range_function(q);
	    double dsrf = short_range_function_derivative(q);
            double ddsrf = short_range_function_second_derivative(q);
	    double dddsrf = short_range_function_third_derivative(q);
            forceD *= ( srf * ( 1.0 + kappa * r1 + kappa * kappa * r2 / 3.0 ) - q * dsrf * ( 1.0 + 2.0 / 3.0 * kappa * r1 ) + q * q / 3.0 * ddsrf );
            Point forceI = muAdotRh * muBdotRh * rh / r4;
            forceI *= ( srf * ( 1.0 + kappa * r1 ) * kappa * kappa * r2 - q * dsrf * ( 3.0 * kappa * r1 + 2.0 ) * kappa * r1 + ddsrf * ( 1.0 + 3.0 * kappa * r1 ) * q * q - q * q * q * dddsrf );
            return ( forceD + forceI ) * std::exp( -kappa * r1);
        } else {
            return {0,0,0};
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
    inline Point dipole_torque(Point mu, Point E) {
        return mu.cross(E);
    }

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
    inline double self_energy(std::array<double,2> m2) const {
      if( self_energy_prefactor.size() != m2.size() )
            throw std::runtime_error("Vectors of self energy prefactors and squared moment are not equal in size!");

      double e_self = 0.0;
      for( int i = 0; i < self_energy_prefactor.size() ; i++ )
            e_self += self_energy_prefactor.at(i) * m2.at(i) * pow(invcutoff,2 * i + 1);
      return e_self;
    }
};

// -------------- Plain ---------------

/**
 * @brief No truncation scheme
 */
struct Plain : public SchemeBase {
    double debye_length;        //!< Debye-length (optional)
    inline Plain( double debye_length=infty ) : SchemeBase(TruncationScheme::plain, infty, debye_length), debye_length(debye_length) {
        name = "plain";
        doi = "Premier mémoire sur l’électricité et le magnétisme by Charles-Augustin de Coulomb"; // :P
        self_energy_prefactor = {0.0, 0.0};
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
    };
    inline double short_range_function(double q) const override { return 1.0; };
    inline double short_range_function_derivative(double q) const { return 0.0; }
    inline double short_range_function_second_derivative(double q) const { return 0.0; }
    inline double short_range_function_third_derivative(double q) const { return 0.0; }
#ifdef NLOHMANN_JSON_HPP
  private:
    inline void _to_json(nlohmann::json &) const override {}
    inline void _from_json(const nlohmann::json &) override {}
#endif
};

#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("[CoulombGalore] plain") {
    using doctest::Approx;
    double cutoff = 29.0;  // cutoff distance
    double zA = 2.0; // charge
    double zB = 3.0; // charge
    Point muA = {19, 7, 11};  // dipole moment
    Point muB = {13, 17, 5};  // dipole moment
    Point r = {23, 0, 0};  // distance vector
    Point rh = {1, 0, 0};  // normalized distance vector
    PairPotential<Plain> pot;

    // Test short-ranged function
    CHECK(pot.short_range_function(0.5) == Approx(1.0));
    CHECK(pot.short_range_function_derivative(0.5) == Approx(0.0));
    CHECK(pot.short_range_function_second_derivative(0.5) == Approx(0.0));
    CHECK(pot.short_range_function_third_derivative(0.5) == Approx(0.0));

    // Test potentials
    CHECK(pot.ion_potential(zA, cutoff + 1.0) == Approx(0.06666666667));
    CHECK(pot.ion_potential(zA, r.norm()) == Approx(0.08695652174));
    CHECK(pot.dipole_potential(muA, ( cutoff + 1.0 ) * rh) == Approx(0.02111111111));
    CHECK(pot.dipole_potential(muA, r) == Approx(0.03591682420));

    // Test fields
    CHECK(pot.ion_field(zA, ( cutoff + 1.0 ) * rh).norm() == Approx(0.002222222222));
    Point E_ion = pot.ion_field(zA, r);
    CHECK(E_ion[0] == Approx(0.003780718336));
    CHECK(E_ion.norm() == Approx(0.003780718336));
    CHECK(pot.dipole_field(muA, ( cutoff + 1.0 ) * rh).norm() == Approx(0.001487948846));
    Point E_dipole = pot.dipole_field(muA, r);
    CHECK(E_dipole[0] == Approx(0.003123202104));
    CHECK(E_dipole[1] == Approx(-0.0005753267034));
    CHECK(E_dipole[2] == Approx(-0.0009040848196));

    // Test energies
    CHECK(pot.ion_ion_energy(zA, zB, ( cutoff + 1.0 )) == Approx(0.2));
    CHECK(pot.ion_ion_energy(zA, zB, r.norm()) == Approx(0.2608695652));
    CHECK(pot.ion_dipole_energy(zA, muB, ( cutoff + 1.0 ) * rh) == Approx(-0.02888888889));
    CHECK(pot.ion_dipole_energy(zA, muB, r) == Approx(-0.04914933837));
    CHECK(pot.dipole_dipole_energy(muA, muB, ( cutoff + 1.0 ) * rh) == Approx(-0.01185185185));
    CHECK(pot.dipole_dipole_energy(muA, muB, r) == Approx(-0.02630064930));

    // Test forces
    CHECK(pot.ion_ion_force(zA, zB, ( cutoff + 1.0 ) * rh).norm() == Approx(0.006666666667));
    Point F_ionion = pot.ion_ion_force(zA, zB, r);
    CHECK(F_ionion[0] == Approx(0.01134215501));
    CHECK(F_ionion.norm() == Approx(0.01134215501));
    CHECK(pot.ion_dipole_force(zB, muA, ( cutoff + 1.0 ) * rh).norm() == Approx(0.004463846540));
    Point F_iondipole = pot.ion_dipole_force(zB, muA, r);
    CHECK(F_iondipole[0] == Approx(0.009369606312));
    CHECK(F_iondipole[1] == Approx(-0.001725980110));
    CHECK(F_iondipole[2] == Approx(-0.002712254459));
    CHECK(pot.dipole_dipole_force(muA, muB, ( cutoff + 1.0 ) * rh).norm() == Approx(0.002129033733));
    Point F_dipoledipole = pot.dipole_dipole_force(muA, muB, r);
    CHECK(F_dipoledipole[0] == Approx(0.003430519474));
    CHECK(F_dipoledipole[1] == Approx(-0.004438234569));
    CHECK(F_dipoledipole[2] == Approx(-0.002551448858));

    // Approximate dipoles by two charges respectively and compare to point-dipoles
    double d = 1e-3; // small distance

    Point r_muA_1 = muA / muA.norm() * d;     // a small distance from dipole A ( the origin )
    Point r_muA_2 = - muA / muA.norm() * d;   // a small distance from dipole B ( the origin )
    Point r_muB_1 = r + muB / muB.norm() * d; // a small distance from dipole B ( 'r' )
    Point r_muB_2 = r - muB / muB.norm() * d; // a small distance from dipole B ( 'r' )
    double z_muA_1 =  muA.norm() / ( 2.0 * d ); // charge 1 of approximative dipole A
    double z_muA_2 = -muA.norm() / ( 2.0 * d ); // charge 2 of approximative dipole A
    double z_muB_1 =  muB.norm() / ( 2.0 * d ); // charge 1 of approximative dipole B
    double z_muB_2 = -muB.norm() / ( 2.0 * d ); // charge 2 of approximative dipole B

    Point muA_approx = r_muA_1 * z_muA_1 + r_muA_2 * z_muA_2;
    Point muB_approx = r_muB_1 * z_muB_1 + r_muB_2 * z_muB_2;

    Point r_z1r = r - r_muA_1; // distance from charge 1 of dipole A to 'r'
    Point r_z2r = r - r_muA_2; // distance from charge 2 of dipole A to 'r'

    // Check that dipole moment of the two charges corresponds to that from the dipole
    CHECK(muA[0] == Approx(muA_approx[0]));
    CHECK(muA[1] == Approx(muA_approx[1]));
    CHECK(muA[2] == Approx(muA_approx[2]));
    CHECK(muB[0] == Approx(muB_approx[0]));
    CHECK(muB[1] == Approx(muB_approx[1]));
    CHECK(muB[2] == Approx(muB_approx[2]));

    // Check potentials
    double potA = pot.dipole_potential(muA, r);
    double potA_1 = pot.ion_potential(z_muA_1, r_z1r.norm());
    double potA_2 = pot.ion_potential(z_muA_2, r_z2r.norm());
    CHECK(potA == Approx(potA_1 + potA_2));

    // Check fields
    Point fieldA = pot.dipole_field(muA, r);
    Point fieldA_1 = pot.ion_field(z_muA_1, r_z1r);
    Point fieldA_2 = pot.ion_field(z_muA_2, r_z2r);
    CHECK(fieldA[0] == Approx(fieldA_1[0] + fieldA_2[0]));
    CHECK(fieldA[1] == Approx(fieldA_1[1] + fieldA_2[1]));
    CHECK(fieldA[2] == Approx(fieldA_1[2] + fieldA_2[2]));

    // Check energies
    double EA = pot.ion_dipole_energy(zB, muA, -r);
    double EA_1 = pot.ion_ion_energy(zB, z_muA_1, r_z1r.norm());
    double EA_2 = pot.ion_ion_energy(zB, z_muA_2, r_z2r.norm());
    CHECK(EA == Approx(EA_1 + EA_2));

    // Check forces
    Point F_ionion_11 = pot.ion_ion_force(z_muA_1, z_muB_1, r_muA_1 - r_muB_1);
    Point F_ionion_12 = pot.ion_ion_force(z_muA_1, z_muB_2, r_muA_1 - r_muB_2);
    Point F_ionion_21 = pot.ion_ion_force(z_muA_2, z_muB_1, r_muA_2 - r_muB_1);
    Point F_ionion_22 = pot.ion_ion_force(z_muA_2, z_muB_2, r_muA_2 - r_muB_2);
    Point F_dipoledipole_approx = F_ionion_11 + F_ionion_12 + F_ionion_21 + F_ionion_22;
    CHECK(F_dipoledipole[0] == Approx(F_dipoledipole_approx[0]));
    CHECK(F_dipoledipole[1] == Approx(F_dipoledipole_approx[1]));
    CHECK(F_dipoledipole[2] == Approx(F_dipoledipole_approx[2]));

    // Check Yukawa-interactions
    double debye_length = 23.0;
    PairPotential<Plain> potY(debye_length);

    // Test potentials
    CHECK(potY.ion_potential(zA, cutoff + 1.0) == Approx(0.01808996296));
    CHECK(potY.ion_potential(zA, r.norm()) == Approx(0.03198951663));
    CHECK(potY.dipole_potential(muA, ( cutoff + 1.0 ) * rh) == Approx(0.01320042949));
    CHECK(potY.dipole_potential(muA, r) == Approx(0.02642612243));

    // Test fields
    CHECK(potY.ion_field(zA, ( cutoff + 1.0 ) * rh).norm() == Approx(0.001389518894));
    Point E_ion_Y = potY.ion_field(zA, r);
    CHECK(E_ion_Y[0] == Approx(0.002781697098));
    CHECK(E_ion_Y.norm() == Approx(0.002781697098));
    CHECK(potY.dipole_field(muA, ( cutoff + 1.0 ) * rh).norm() == Approx(0.001242154748));
    Point E_dipole_Y = potY.dipole_field(muA, r);
    CHECK(E_dipole_Y[0] == Approx(0.002872404612));
    CHECK(E_dipole_Y[1] == Approx(-0.0004233017324));
    CHECK(E_dipole_Y[2] == Approx(-0.0006651884364));

    // Test forces
    CHECK(potY.dipole_dipole_force(muA, muB, ( cutoff + 1.0 ) * rh).norm() == Approx(0.001859094075));
    Point F_dipoledipole_Y = potY.dipole_dipole_force(muA, muB, r);
    CHECK(F_dipoledipole_Y[0] == Approx(0.003594120919));
    CHECK(F_dipoledipole_Y[1] == Approx(-0.003809715590));
    CHECK(F_dipoledipole_Y[2] == Approx(-0.002190126354));
}

#endif

// -------------- Ewald real-space ---------------

/**
 * @brief Ewald real-space scheme
 */
struct Ewald : public SchemeBase {
    double alpha;               //!< Damping-parameter
    double alphaRed, alphaRed2; //!< Reduced damping-parameter, and squared
    double eps_sur;             //!< Dielectric constant of the surrounding medium
    double debye_length;        //!< Debye-length (optional)
    const double pi_sqrt = 2.0*std::sqrt(std::atan(1.0));

    /**
     * @param cutoff distance cutoff
     * @param alpha damping-parameter
     */
    inline Ewald(double cutoff, double alpha, double eps_sur, double debye_length=infty) : SchemeBase(TruncationScheme::ewald, cutoff), alpha(alpha), eps_sur(eps_sur), debye_length(debye_length) {
        name = "Ewald real-space";
        alphaRed = alpha*cutoff;
        alphaRed2 = alphaRed * alphaRed;
        self_energy_prefactor = { - alphaRed / pi_sqrt, -pow(alphaRed,3) * 2.0 / 3.0 / pi_sqrt };
        if ( eps_sur < 1.0 )
            throw std::runtime_error("Dielectric constant of the surrounding medium is less than one");
        T0 = 2.0 * (eps_sur - 1.0 ) / ( 2.0 * eps_sur + 1.0 );
    }

    inline double short_range_function(double q) const override { return std::erfc(alphaRed*q); }
    inline double short_range_function_derivative(double q) const { return -2.0 * std::exp( - alphaRed2 * q * q ) * alphaRed / pi_sqrt; }
    inline double short_range_function_second_derivative(double q) const { return 4.0 * std::exp( - alphaRed2 * q * q ) * alphaRed2 * alphaRed * q / pi_sqrt; }
    inline double short_range_function_third_derivative(double q) const { return -8.0 * std::exp( - alphaRed2 * q * q ) * alphaRed2 * alphaRed * ( alphaRed2 * q * q - 0.5 ) / pi_sqrt; }

#ifdef NLOHMANN_JSON_HPP
  private:
    inline void _from_json(const nlohmann::json &j) override { alpha = j.at("alpha").get<int>(); }
    inline void _to_json(nlohmann::json &j) const override { j = {{"alpha", alpha}}; }
#endif
};

#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("[CoulombGalore] Ewald real-space") {
    using doctest::Approx;
    double cutoff = 29.0;  // cutoff distance
    double alpha = 0.1; // damping-parameter
    double eps_sur = infty;
    PairPotential<Ewald> pot(cutoff,alpha,eps_sur);
    // Test short-ranged function
    CHECK(pot.short_range_function(0.5) == Approx(0.04030497436));
    CHECK(pot.short_range_function_derivative(0.5) == Approx(-0.399713585));
    CHECK(pot.short_range_function_second_derivative(0.5) == Approx(3.36159125));
    CHECK(pot.short_range_function_third_derivative(0.5) == Approx(-21.54779991));
}

#endif

// -------------- Wolf ---------------

/**
 * @brief Wolf scheme
 */
struct Wolf : public SchemeBase {
    double alpha;               //!< Damping-parameter
    double alphaRed, alphaRed2; //!< Reduced damping-parameter, and squared
    const double pi_sqrt = 2.0*std::sqrt(std::atan(1.0));

    /**
     * @param cutoff distance cutoff
     * @param alpha damping-parameter
     */
    inline Wolf(double cutoff, double alpha) : SchemeBase(TruncationScheme::wolf, cutoff), alpha(alpha) {
        name = "Wolf";
        alphaRed = alpha*cutoff;
        alphaRed2 = alphaRed * alphaRed;
        self_energy_prefactor = { - alphaRed / pi_sqrt, -pow(alphaRed,3) * 2.0 / 3.0 / pi_sqrt };
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
    }

    inline double short_range_function(double q) const override { return ( std::erfc(alphaRed*q) - q * std::erfc(alphaRed) ); }
    inline double short_range_function_derivative(double q) const { return ( -2.0 * std::exp( - alphaRed2 * q * q ) * alphaRed / pi_sqrt - std::erfc(alphaRed) ); }
    inline double short_range_function_second_derivative(double q) const { return 4.0 * std::exp( - alphaRed2 * q * q ) * alphaRed2 * alphaRed * q / pi_sqrt; }
    inline double short_range_function_third_derivative(double q) const { return -8.0 * std::exp( - alphaRed2 * q * q ) * alphaRed2 * alphaRed * ( alphaRed2 * q * q - 0.5 ) / pi_sqrt; }

#ifdef NLOHMANN_JSON_HPP
  private:
    inline void _from_json(const nlohmann::json &j) override { alpha = j.at("alpha").get<int>(); }
    inline void _to_json(nlohmann::json &j) const override { j = {{"alpha", alpha}}; }
#endif
};

#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("[CoulombGalore] Wolf") {
    using doctest::Approx;
    double cutoff = 29.0;  // cutoff distance
    double alpha = 0.1; // damping-parameter
    PairPotential<Wolf> pot(cutoff,alpha);
    // Test short-ranged function
    CHECK(pot.short_range_function(0.5) == Approx(0.04028442542));
    CHECK(pot.short_range_function_derivative(0.5) == Approx(-0.3997546829));
    CHECK(pot.short_range_function_second_derivative(0.5) == Approx(3.36159125));
    CHECK(pot.short_range_function_third_derivative(0.5) == Approx(-21.54779991));
    CHECK(pot.short_range_function(1.0) == Approx(0.0));
}

#endif

// -------------- qPotential ---------------

/**
 * @brief qPotential scheme
 */
struct qPotential : public SchemeBase {
    int order; //!< Number of moment to cancel

    /**
     * @param cutoff distance cutoff
     * @param order number of moments to cancel
     */
    inline qPotential(double cutoff, int order) : SchemeBase(TruncationScheme::qpotential, cutoff), order(order) {
        name = "qpotential";
        self_energy_prefactor = {-1.0, -1.0};
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
    }

    inline double short_range_function(double q) const override { return qPochhammerSymbol(q, 0, order); }
    inline double short_range_function_derivative(double q) const { return qPochhammerSymbolDerivative(q, 0, order); }
    inline double short_range_function_second_derivative(double q) const { return qPochhammerSymbolSecondDerivative(q, 0, order); }
    inline double short_range_function_third_derivative(double q) const { return qPochhammerSymbolThirdDerivative(q, 0, order); }

#ifdef NLOHMANN_JSON_HPP
  private:
    inline void _from_json(const nlohmann::json &j) override { order = j.at("order").get<int>(); }
    inline void _to_json(nlohmann::json &j) const override { j = {{"order", order}}; }
#endif
};

#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("[CoulombGalore] qPotential") {
    using doctest::Approx;
    double cutoff = 29.0;  // cutoff distance
    int order = 4; // number of higher order moments to cancel - 1
    PairPotential<qPotential> pot(cutoff,order);
    // Test short-ranged function
    CHECK(pot.short_range_function(0.5) == Approx(0.3076171875));
    CHECK(pot.short_range_function_derivative(0.5) == Approx(-1.453125));
    CHECK(pot.short_range_function_second_derivative(0.5) == Approx(1.9140625));
    CHECK(pot.short_range_function_third_derivative(0.5) == Approx(17.25));
    CHECK(pot.short_range_function(1.0) == Approx(0.0));
    CHECK(pot.short_range_function_derivative(1.0) == Approx(0.0));
    CHECK(pot.short_range_function_second_derivative(1.0) == Approx(0.0));
    CHECK(pot.short_range_function_third_derivative(1.0) == Approx(0.0));
    CHECK(pot.short_range_function(0.0) == Approx(1.0));
    CHECK(pot.short_range_function_derivative(0.0) == Approx(-1.0));
    CHECK(pot.short_range_function_second_derivative(0.0) == Approx(-2.0));
    CHECK(pot.short_range_function_third_derivative(0.0) == Approx(0.0));
}

#endif

// -------------- Poisson ---------------

struct PoissonSimple : public SchemeBase {
    signed int C, D;

    inline PoissonSimple(double cutoff, signed int C, signed int D)
        : SchemeBase(TruncationScheme::poisson, cutoff), C(C), D(D) {
        if ( ( C < 1 ) || ( D < -1 ) )
            throw std::runtime_error("`C` must be larger than zero and `D` must be larger or equal to negative one");
        name = "poisson";
        doi = "10.1088/1367-2630/ab1ec1";
        double a1 = -double(C + D) / double(C);
        self_energy_prefactor = {a1, a1};
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
    }
    inline double short_range_function(double q) const override {
        double tmp = 0;
        for (signed int c = 0; c < C; c++)
            tmp += double( binomial( D - 1 + c , c ) ) * double( C - c ) / double(C) * std::pow( q , double( c ) );
        return std::pow(1.0 - q, double( D + 1 )) * tmp;
    }
    inline double short_range_function_derivative(double q) const {
        double tmp1 = 1.0;
        double tmp2 = 0.0;
        for (signed int c = 1; c < C; c++) {
            tmp1 += double( binomial( D - 1 + c , c ) ) * double( C - c ) / double(C) * std::pow( q , double( c ) );
            tmp2 += double( binomial( D - 1 + c , c ) ) * double( C - c ) / double(C) * double( c ) * std::pow( q , double( c ) - 1.0 );
        }
        return ( -double(D + 1) * pow( 1.0 - q , double( D ) ) * tmp1 + pow( 1.0 - q , double( D + 1 ) ) * tmp2 );
    }
    inline double short_range_function_second_derivative(double q) const { return double( binomial( C + D , C ) * D ) * std::pow( 1.0 - q , double( D ) - 1.0 ) * std::pow( q , double( C ) - 1.0 ); };
    inline double short_range_function_third_derivative(double q) const { return double( binomial( C + D , C ) * D ) * std::pow( 1.0 - q , double( D ) - 2.0 ) * std::pow( q , double( C ) - 2.0 ) * ( ( 2.0 - double(C + D) ) * q +  double( C ) - 1.0 ); };

#ifdef NLOHMANN_JSON_HPP
  private:
    inline void _from_json(const nlohmann::json &j) override {
        C = j.at("C").get<double>();
        D = j.at("D").get<double>();
    }
    inline void _to_json(nlohmann::json &j) const override { j = {{"C", C}, {"D", D}}; }
#endif
};

/**
 * @brief Poisson scheme, also works for Yukawa-potential
 *
 * A general scheme which pending two parameters `C` and `D` can model several different pair-potentials.
 * For infinite Debye-length the following holds:
 *
 *  Type            | `C` | `D` | Reference / Comment
 *  --------------- | --- | --- | ----------------------
 *  `plain`         |  1  | -1  | Plain Coulomb
 *  `wolf`          |  1  |  0  | Undamped Wolf, DOI: 10.1063/1.478738
 *  `fennel`        |  1  |  1  | Levitt ( or undamped Fennell ), DOI: 10.1016/0010-4655(95)00049-L (or 10.1063/1.2206581)
 *  `kale`          |  1  |  2  | Kale, DOI: 10.1021/ct200392u
 *  `mccann`        |  1  |  3  | McCann, DOI: 10.1021/ct300961
 *  `fukuda`        |  2  |  1  | Undamped Fukuda, DOI: 10.1063/1.3582791
 *  `markland`      |  2  |  2  | Markland, DOI: 10.1016/j.cplett.2008.09.019
 *  `stenqvist`     |  3  |  3  | Stenqvist, DOI: 10.1088/1367-2630/ab1ec1
 *  `fanourgakis`   |  4  |  3  | Fanourgakis, DOI: 10.1063/1.3216520
 *
 *  The following keywords are required:
 *
 *  Keyword        |  Description
 *  -------------- |  -------------------------------------------
 *  `cutoff`       |  Spherical cutoff in angstroms
 *  `C`            |  Number of cancelled derivatives at origin -2 (starting from second derivative)
 *  `D`            |  Number of cancelled derivatives at the cut-off (starting from zeroth derivative)
 *  `debye_length` |  Debye-length (optional)
 *
 *  More info:
 *
 *  - http://dx.doi.org/10.1088/1367-2630/ab1ec1
 */
struct Poisson : public SchemeBase {
    signed int C, D;                          //!< Derivative cancelling-parameters
    double debye_length, kappaRed, kappaRed2; //!< Debye-length and reduced inverse Debye-length (optional)
    double yukawa_denom, binomCDC;
    bool yukawa;

    inline Poisson(double cutoff, signed int C, signed int D, double debye_length=infty)
        : SchemeBase(TruncationScheme::poisson, cutoff, debye_length), C(C), D(D), debye_length(debye_length) {
        if ( ( C < 1 ) || ( D < -1 ) )
            throw std::runtime_error("`C` must be larger than zero and `D` must be larger or equal to negative one");
        name = "poisson";
        doi = "10.1088/1367-2630/ab1ec1";
        double a1 = -double(C + D) / double(C);
        kappaRed = cutoff / debye_length;
        yukawa = false;
        if( std::fabs( kappaRed ) > 1e-6 ) {
            yukawa = true;
            a1 *= -2.0 * kappaRed / ( 1.0 - std::exp( 2.0 * kappaRed) );
            kappaRed2 = kappaRed * kappaRed;
            yukawa_denom = 1.0 / ( 1.0 - std::exp( 2.0 * kappaRed ) );
        }
        binomCDC = double( binomial( C + D , C ) * D );
        self_energy_prefactor = {a1, a1};
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
    }
    inline double short_range_function(double q) const override {
        double tmp = 0;
        double qp = q;
        if( yukawa )
            qp = ( 1.0 - std::exp( 2.0 * kappaRed * q ) ) * yukawa_denom;
        for (signed int c = 0; c < C; c++)
            tmp += double( binomial( D - 1 + c , c ) ) * double( C - c ) / double(C) * std::pow( qp , double( c ) );
        return std::pow(1.0 - qp, double( D + 1 )) * tmp;
    }
    inline double short_range_function_derivative(double q) const {
        double qp = q;
        double dqpdq = 1.0;
        if( yukawa ) {
            qp = ( 1.0 - std::exp( 2.0 * kappaRed * q ) ) * yukawa_denom;
            dqpdq = -2.0 * kappaRed * std::exp( 2.0 * kappaRed * q ) * yukawa_denom;
        }
        double tmp1 = 1.0;
        double tmp2 = 0.0;
        for (signed int c = 1; c < C; c++) {
            tmp1 += double( binomial( D - 1 + c , c ) ) * double( C - c ) / double(C) * std::pow( qp , double( c ) );
            tmp2 += double( binomial( D - 1 + c , c ) ) * double( C - c ) / double(C) * double( c ) * std::pow( qp , double( c ) - 1.0 );
        }
        double dSdqp = ( -double(D + 1) * pow( 1.0 - qp , double( D ) ) * tmp1 + pow( 1.0 - qp , double( D + 1 ) ) * tmp2 );
        return dSdqp * dqpdq;
    }
    inline double short_range_function_second_derivative(double q) const {
        double qp = q;
        double dqpdq = 1.0;
        double d2qpdq2 = 0.0;
        double dSdqp = 0.0;
        if( yukawa ) {
            qp = ( 1.0 - std::exp( 2.0 * kappaRed * q ) ) * yukawa_denom;
            dqpdq = -2.0 * kappaRed * std::exp( 2.0 * kappaRed * q ) * yukawa_denom;
            d2qpdq2 = -4.0 * kappaRed2 * std::exp( 2.0 * kappaRed * q ) * yukawa_denom;
            double tmp1 = 1.0;
            double tmp2 = 0.0;
            for (signed int c = 1; c < C; c++) {
                tmp1 += double( binomial( D - 1 + c , c ) ) * double( C - c ) / double(C) * std::pow( qp , double( c ) );
                tmp2 += double( binomial( D - 1 + c , c ) ) * double( C - c ) / double(C) * double( c ) * std::pow( qp , double( c ) - 1.0 );
            }
            dSdqp = ( -double(D + 1) * pow( 1.0 - qp , double( D ) ) * tmp1 + pow( 1.0 - qp , double( D + 1 ) ) * tmp2 );
        }
        double d2Sdqp2 = binomCDC * std::pow( 1.0 - qp , double( D ) - 1.0 ) * std::pow( qp , double( C ) - 1.0 );
        return ( d2Sdqp2 * dqpdq * dqpdq + dSdqp * d2qpdq2 );
    };
    inline double short_range_function_third_derivative(double q) const {
        double qp = q;
        double dqpdq = 1.0;
        double d2qpdq2 = 0.0;
        double d3qpdq3 = 0.0;
        double d2Sdqp2 = 0.0;
        double dSdqp = 0.0;
        if( yukawa ) {
            qp = ( 1.0 - std::exp( 2.0 * kappaRed * q ) ) * yukawa_denom;
            dqpdq = -2.0 * kappaRed * std::exp( 2.0 * kappaRed * q ) * yukawa_denom;
            d2qpdq2 = -4.0 * kappaRed2 * std::exp( 2.0 * kappaRed * q ) * yukawa_denom;
            d3qpdq3 = -8.0 * kappaRed2 * kappaRed * std::exp( 2.0 * kappaRed * q ) * yukawa_denom;
            d2Sdqp2 = binomCDC * std::pow( 1.0 - qp , double( D ) - 1.0 ) * std::pow( qp , double( C ) - 1.0 );
            double tmp1 = 1.0;
            double tmp2 = 0.0;
            for (signed int c = 1; c < C; c++) {
                tmp1 += double( binomial( D - 1 + c , c ) ) * double( C - c ) / double(C) * std::pow( qp , double( c ) );
                tmp2 += double( binomial( D - 1 + c , c ) ) * double( C - c ) / double(C) * double( c ) * std::pow( qp , double( c ) - 1.0 );
            }
            dSdqp = ( -double(D + 1) * pow( 1.0 - qp , double( D ) ) * tmp1 + pow( 1.0 - qp , double( D + 1 ) ) * tmp2 );
        }
        double d3Sdqp3 = binomCDC * std::pow( 1.0 - qp , double( D ) - 2.0 ) * std::pow( qp , double( C ) - 2.0 ) * ( ( 2.0 - double(C + D) ) * qp +  double( C ) - 1.0 );
        return ( d3Sdqp3 * dqpdq * dqpdq * dqpdq + 3.0*d2Sdqp2 * dqpdq * d2qpdq2 + dSdqp * d3qpdq3 );
    };

#ifdef NLOHMANN_JSON_HPP
  private:
    inline void _from_json(const nlohmann::json &j) override {
        C = j.at("C").get<double>();
        D = j.at("D").get<double>();
    }
    inline void _to_json(nlohmann::json &j) const override { j = {{"C", C}, {"D", D}}; }
#endif
};

#ifdef DOCTEST_LIBRARY_INCLUDED

TEST_CASE("[CoulombGalore] Poisson") {
    using doctest::Approx;
    signed C = 3; // number of cancelled derivatives at origin -2 (starting from second derivative)
    signed D = 3; // number of cancelled derivatives at the cut-off (starting from zeroth derivative)
    double cutoff = 29.0;  // cutoff distance
    PairPotential<Poisson> pot33(cutoff,C,D);

    // Test short-ranged function
    CHECK(pot33.short_range_function(0.5) == Approx(0.15625));
    CHECK(pot33.short_range_function_derivative(0.5) == Approx(-1.0));
    CHECK(pot33.short_range_function_second_derivative(0.5) == Approx(3.75));
    CHECK(pot33.short_range_function_third_derivative(0.5) == Approx(0.0));
    CHECK(pot33.short_range_function_third_derivative(0.6) == Approx(-5.76));
    CHECK(pot33.short_range_function(1.0) == Approx(0.0));
    CHECK(pot33.short_range_function_derivative(1.0) == Approx(0.0));
    CHECK(pot33.short_range_function_second_derivative(1.0) == Approx(0.0));
    CHECK(pot33.short_range_function_third_derivative(1.0) == Approx(0.0));
    CHECK(pot33.short_range_function(0.0) == Approx(1.0));
    CHECK(pot33.short_range_function_derivative(0.0) == Approx(-2.0));
    CHECK(pot33.short_range_function_second_derivative(0.0) == Approx(0.0));
    CHECK(pot33.short_range_function_third_derivative(0.0) == Approx(0.0));

    C = 4; // number of cancelled derivatives at origin -2 (starting from second derivative)
    D = 3; // number of cancelled derivatives at the cut-off (starting from zeroth derivative)
    double zA = 2.0; // charge
    double zB = 3.0; // charge
    Point muA = {19, 7, 11};  // dipole moment
    Point muB = {13, 17, 5};  // dipole moment
    Point r = {23, 0, 0};  // distance vector
    Point rh = {1, 0, 0};  // normalized distance vector
    PairPotential<Poisson> pot43(cutoff,C,D);

    // Test short-ranged function
    CHECK(pot43.short_range_function(0.5) == Approx(0.19921875));
    CHECK(pot43.short_range_function_derivative(0.5) == Approx(-1.1484375));
    CHECK(pot43.short_range_function_second_derivative(0.5) == Approx(3.28125));
    CHECK(pot43.short_range_function_third_derivative(0.5) == Approx(6.5625));

    // Test potentials
    CHECK(pot43.ion_potential(zA, cutoff) == Approx(0.0));
    CHECK(pot43.ion_potential(zA, r.norm()) == Approx(0.0009430652121));
    CHECK(pot43.dipole_potential(muA, cutoff * rh) == Approx(0.0));
    CHECK(pot43.dipole_potential(muA, r) == Approx(0.005750206554));

    // Test fields
    CHECK(pot43.ion_field(zA, cutoff * rh).norm() == Approx(0.0));
    Point E_ion = pot43.ion_field(zA, r);
    CHECK(E_ion[0] == Approx(0.0006052849004));
    CHECK(E_ion.norm() == Approx(0.0006052849004));
    CHECK(pot43.dipole_field(muA, cutoff * rh).norm() == Approx(0.0));
    Point E_dipole = pot43.dipole_field(muA, r);
    CHECK(E_dipole[0] == Approx(0.002702513754));
    CHECK(E_dipole[1] == Approx(-0.00009210857180));
    CHECK(E_dipole[2] == Approx(-0.0001447420414));

    // Test energies
    CHECK(pot43.ion_ion_energy(zA, zB, cutoff) == Approx(0.0));
    CHECK(pot43.ion_ion_energy(zA, zB, r.norm()) == Approx(0.002829195636));
    CHECK(pot43.ion_dipole_energy(zA, muB, cutoff * rh) == Approx(0.0));
    CHECK(pot43.ion_dipole_energy(zA, muB, r) == Approx(-0.007868703705));
    CHECK(pot43.dipole_dipole_energy(muA, muB, cutoff * rh) == Approx(0.0));
    CHECK(pot43.dipole_dipole_energy(muA, muB, r) == Approx(-0.03284312288));

    // Test forces
    CHECK(pot43.ion_ion_force(zA, zB, cutoff * rh).norm() == Approx(0.0));
    Point F_ionion = pot43.ion_ion_force(zA, zB, r);
    CHECK(F_ionion[0] == Approx(0.001815854701));
    CHECK(F_ionion.norm() == Approx(0.001815854701));
    CHECK(pot43.ion_dipole_force(zB, muA, cutoff * rh).norm() == Approx(0.0));
    Point F_iondipole = pot43.ion_dipole_force(zB, muA, r);
    CHECK(F_iondipole[0] == Approx(0.008107541263));
    CHECK(F_iondipole[1] == Approx(-0.0002763257154));
    CHECK(F_iondipole[2] == Approx(-0.0004342261242));
    CHECK(pot43.dipole_dipole_force(muA, muB, cutoff * rh).norm() == Approx(0.0));
    Point F_dipoledipole = pot43.dipole_dipole_force(muA, muB, r);
    CHECK(F_dipoledipole[0] == Approx(0.009216400961));
    CHECK(F_dipoledipole[1] == Approx(-0.002797126801));
    CHECK(F_dipoledipole[2] == Approx(-0.001608010094));

    // Test Yukawa-interactions
    C = 3; // number of cancelled derivatives at origin -2 (starting from second derivative)
    D = 3; // number of cancelled derivatives at the cut-off (starting from zeroth derivative)
    cutoff = 29.0;  // cutoff distance
    double debye_length = 23.0;
    PairPotential<Poisson> potY(cutoff,C,D,debye_length);

    // Test short-ranged function
    CHECK(potY.short_range_function(0.5) == Approx(0.5673222034));
    CHECK(potY.short_range_function_derivative(0.5) == Approx(-1.437372757));
    CHECK(potY.short_range_function_second_derivative(0.5) == Approx(-2.552012334));
    CHECK(potY.short_range_function_third_derivative(0.5) == Approx(4.384434209));

    // Test potentials
    CHECK(potY.ion_potential(zA, cutoff) == Approx(0.0));
    CHECK(potY.ion_potential(zA, r.norm()) == Approx(0.003344219306));
    CHECK(potY.dipole_potential(muA, cutoff * rh) == Approx(0.0));
    CHECK(potY.dipole_potential(muA, r) == Approx(0.01614089171));

    // Test fields
    CHECK(potY.ion_field(zA, cutoff * rh).norm() == Approx(0.0));
    Point E_ion_Y = potY.ion_field(zA, r);
    CHECK(E_ion_Y[0] == Approx(0.001699041230));
    CHECK(E_ion_Y.norm() == Approx(0.001699041230));
    CHECK(potY.dipole_field(muA, cutoff * rh).norm() == Approx(0.0));
    Point E_dipole_Y = potY.dipole_field(muA, r);
    CHECK(E_dipole_Y[0] == Approx(0.004956265485));
    CHECK(E_dipole_Y[1] == Approx(-0.0002585497523));
    CHECK(E_dipole_Y[2] == Approx(-0.0004062924688));

    // Test forces
    CHECK(potY.dipole_dipole_force(muA, muB, cutoff * rh).norm() == Approx(0.0));
    Point F_dipoledipole_Y = potY.dipole_dipole_force(muA, muB, r);
    CHECK(F_dipoledipole_Y[0] == Approx(0.002987655338));
    CHECK(F_dipoledipole_Y[1] == Approx(-0.005360251621));
    CHECK(F_dipoledipole_Y[2] == Approx(-0.003081497308));
}

#endif

// -------------- Fanourgakis ---------------

/**
 * @brief Fanourgakis scheme.
 * @note This is the same as using the 'Poisson' approach with parameters 'C=4' and 'D=3'
 */
struct Fanourgakis : public SchemeBase {
    /**
     * @param cutoff distance cutoff
     */
    inline Fanourgakis(double cutoff) : SchemeBase(TruncationScheme::qpotential, cutoff) {
        name = "fanourgakis";
        doi = "10.1063/1.3216520";
        self_energy_prefactor = {-1.0, -1.0};
        T0 = short_range_function_derivative(1.0) - short_range_function(1.0) + short_range_function(0.0);
    }

    inline double short_range_function(double q) const override { return pow(1.0 - q,4.0) * ( 1.0 + 2.25 * q + 3.0 * q*q + 2.5 * q*q*q ); }
    inline double short_range_function_derivative(double q) const { return ( -1.75 + 26.25 * pow(q,4.0) - 42.0 * pow(q,5.0) + 17.5 * pow(q,6.0) ); }
    inline double short_range_function_second_derivative(double q) const { return 105.0 * pow(q,3.0) * pow(q - 1.0, 2); };
    inline double short_range_function_third_derivative(double q) const { return 525.0 * pow(q,2.0) * (q - 0.6) * ( q - 1.0); };

#ifdef NLOHMANN_JSON_HPP
  private:
    inline void _to_json(nlohmann::json &) const override {}
    inline void _from_json(const nlohmann::json &) override {}
#endif
};

#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("[CoulombGalore] Fanourgakis") {
    using doctest::Approx;
    double cutoff = 29.0;  // cutoff distance
    PairPotential<Fanourgakis> pot(cutoff);
    // Test short-ranged function
    CHECK(pot.short_range_function(0.5) == Approx(0.1992187500));
    CHECK(pot.short_range_function_derivative(0.5) == Approx(-1.1484375));
    CHECK(pot.short_range_function_second_derivative(0.5) == Approx(3.28125));
    CHECK(pot.short_range_function_third_derivative(0.5) == Approx(6.5625));
}
#endif


} // namespace CoulombGalore
