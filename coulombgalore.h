#pragma once

#include <string>
#include <limits>
#include <cmath>
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

/**
 * @brief Help-function for the q-potential
 *
 * More information here: http://mathworld.wolfram.com/q-PochhammerSymbol.html
 * P = 300 gives an error of about 10^-17 for k < 4
 */
inline double qPochhammerSymbol(double q, int k = 1, int P = 300) {
    double value = 1.0;
    double temp = std::pow(q, k);
    for (int i = 0; i < P; i++) {
        value *= (1.0 - temp);
        temp *= q;
    }
    return value;
}

#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("qPochhammerSymbol") {
    double q = 0.5;
    CHECK(qPochhammerSymbol(q, 0, 0) == 1);
    CHECK(qPochhammerSymbol(0, 0, 1) == 0);
    CHECK(qPochhammerSymbol(1, 0, 1) == 0);
    CHECK(qPochhammerSymbol(1, 1, 2) == 0);
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
    std::array<double> self_energy_prefactor; //!< Prefactor for self-energies
    inline SchemeBase(TruncationScheme scheme, double cutoff) : scheme(scheme), cutoff(cutoff) {}

    /**
     * @brief Splitting function
     * @param q q=r/Rcutoff
     * @todo How should this be expanded to higher order moments?
     */
    virtual double splitting_function(double) const = 0;

    virtual double splitting_function_derivative(double q, double dh=1e-6) {
        return ( splitting_function(q + dh) - splitting_function(q - dh) ) / ( 2 * dh );
    }

    virtual double splitting_function_second_derivative(double q, double dh=1e-6) {
        return ( splitting_function_derivative(q + dh , dh ) - splitting_function_derivative(q - dh , dh ) ) / ( 2 * dh );
    }

    virtual double splitting_function_third_derivative(double q, double dh=1e-6) {
        return ( splitting_function_second_derivative(q + dh , dh ) - splitting_function_second_derivative(q - dh , dh ) ) / ( 2 * dh );
    }

    /**
     * @brief Calculate dielectric constant
     * @param M2V system dipole moment fluctuation
     */
    virtual double calc_dielectric(double) const = 0;

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
    double cutoff2; // square cutoff distance
  public:
    using Tscheme::cutoff;
    using Tscheme::self_energy_prefactor;
    using Tscheme::splitting_function;
    using Tscheme::splitting_function_derivative;
    using Tscheme::splitting_function_second_derivative;
    using Tscheme::splitting_function_third_derivative;

    template <class... Args> PairPotential(Args &&... args) : Tscheme(args...) {
      invcutoff = 1.0 / cutoff;
      cutoff2 = cutoff*cutoff;
    }

    /**
     * @brief ion potential
     * @returns potential from ion in electrostatic units ( why not Hartree atomic units? )
     * @param z charge
     * @param r charge separation
     *
     * @details The potential from a charge is described by
     * @f[
     *     \Phi({\bf r},z) = \frac{z}{|{\bf r}|}s(q)
     * @f]
     *
     */
    inline double ion_potential(double z, double r) {
        if ( r < cutoff )
            return z / r * splitting_function(r * invcutoff);
        else
            return 0.0;
    }

    /**
     * @brief dipole potential
     * @returns potential from dipole in electrostatic units ( why not Hartree atomic units? )
     * @param mu dipole
     * @param r distance-vector from dipole
     *
     * @details The potential from a charge is described by
     * @f[
     *     \Phi({\bf r},{\bf \mu}) = \frac{{\bf \mu} \cdot \hat{{\bf r}} }{|{\bf r}|^2} \left( s(q) - qs^{\prime}(q) \right)
     * @f]
     *
     */
    inline double dipole_potential(Point mu, Point r) {
        double r2 = r.squaredNorm();
        if ( r2 < cutoff2 ) {
            double r1 = std::sqrt(r2);
            double q = r1 * invcutoff;
            return mu.dot(r) / r2 / r1  * ( splitting_function(q) - q * splitting_function_derivative(q) );
        } else {
            return 0.0;
        }
    }

    /**
     * @brief field from ion
     * @returns field in electrostatic units ( why not Hartree atomic units? )
     * @param z charge
     * @param r distance-vector to charge
     *
     * @details The field from a charge is described by
     * @f[
     *     {\bf E}({\bf r},z) = -\nabla \Phi({\bf r},z) = \frac{z \hat{{\bf r}} }{|{\bf r}|^2} \left( s(q) - qs^{\prime}(q) \right)
     * @f]
     *
     */
    inline Point ion_field(double z, Point r) {
        double r2 = r.squaredNorm();
        if ( r2 < cutoff2 ) {
            double r1 = std::sqrt(r2);
            double q = r1 * invcutoff;
            return z * r / r2 / r1 * ( splitting_function(q) - q * splitting_function_derivative(q) );
        } else {
            return {0,0,0};
        }
    }

    /**
     * @brief field from dipole
     * @returns field in electrostatic units ( why not Hartree atomic units? )
     * @param mu dipole
     * @param r distance-vector to dipole
     * @note not finished
     *
     * @details The field from a dipole is described by
     * @f[
     *     {\bf E}({\bf r},{\bf \mu}) = -\nabla \Phi({\bf r},{\bf \mu}) = \frac{3 ( {\bf \mu} \cdot \hat{{\bf r}} ) \hat{{\bf r}} - \hat{{\bf \mu}} }{|{\bf r}|^3}
     * @f]
     *
     */
    inline Point dipole_field(Point mu, Point r) {
        double r2 = r.squaredNorm();
        if ( r2 < cutoff2 ) {
            double r1 = std::sqrt(r2);
            double q = r1 * invcutoff;
            double second_derivative_scaled = q * q / 3.0 * splitting_function_second_derivative(q);
            Point field = ( 3.0 * mu.dot(r) * r / r2 - mu ) / r2 / r1 * ( splitting_function(q) - q * splitting_function_derivative(q) + second_derivative_scaled );
            field += mu / r2 / r1 * second_derivative_scaled;
            return field;
        } else {
            return {0,0,0};
        }
    }

    /**
     * @brief ion-ion interaction force
     * @returns interaction force in electrostatic units ( why not Hartree atomic units? )
     * @param zA charge
     * @param zB charge
     * @param r distance-vector between charges
     */
    inline Point ion_ion_force(double zA, double zB, Point r) {
        return ion_field(zA,r)*zB;
    }

    /**
     * @brief ion-dipole interaction force
     * @returns interaction force in electrostatic units ( why not Hartree atomic units? )
     * @param z charge
     * @param mu dipole moment
     * @param r distance-vector between dipole and charge, mu - z
     */
    inline Point ion_dipole_force(double z, Point mu, Point r) {
        return dipole_field(mu,r)*z;
    }

    /**
     * @brief dipole-dipole interaction energy
     * @returns interaction energy in electrostatic units ( why not Hartree atomic units? )
     * @param muA dipole moment of particle A
     * @param muB dipole moment of particle B
     * @param r distance-vector between dipoles
     * @note not finished
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
            Point force = 3.0 * ( ( 5.0* muAdotRh*muBdotRh - muA.dot(muB) ) * rh - muBdotRh * muA + muAdotRh * muB  ) / r4;
            double second_derivative = splitting_function_second_derivative(q);
            force *= ( splitting_function(q) - q * splitting_function_derivative(q) + q * q / 3.0 * second_derivative );
            force += muAdotRh * muBdotRh * rh / r4 *( second_derivative - q * splitting_function_third_derivative(q) ) * q * q;
            return force;
        } else {
            return {0,0,0};
        }
    }

    /**
     * @brief ion-ion interaction energy
     * @returns interaction energy in electrostatic units ( why not Hartree atomic units? )
     * @param zA charge
     * @param zB charge
     * @param r charge separation
     */
    inline double ion_ion_energy(double zA, double zB, double r) {
        return ion_potential(zA,r)*zB;
    }

    /**
     * @brief ion-dipole interaction energy
     * @returns interaction energy in electrostatic units ( why not Hartree atomic units? )
     * @param z charge
     * @param mu dipole moment
     * @param r distance-vector between dipole and charge, mu - z
     * @note the direction of r is from charge towards dipole
     */
    inline double ion_dipole_energy(double z, Point mu, Point r) {
        return -mu.dot(ion_field(z,r));
    }

    /**
     * @brief dipole-dipole interaction energy
     * @returns interaction energy in electrostatic units ( why not Hartree atomic units? )
     * @param muA dipole moment of particle A
     * @param muB dipole moment of particle B
     * @param r distance-vector between dipoles
     */
    inline double dipole_dipole_energy(Point muA, Point muB, Point r) {
        return -muA.dot(dipole_field(muB,r));
    }

    /**
     * @brief self-energy for all type of interactions
     * @param zz charge product
     * @returns self energy in electrostatic units ( why not Hartree atomic units? )
     * @param mumu product between dipole moment scalars
     */
    inline double self_energy(std::array<double> m2) const {
      if( self_energy_prefactor.size() != m2.size() )
            throw std::runtime_error("Vectors of self energy prefactors and squared moment are not equal in size!");

      double e_self = 0.0;
      for( int i = 0; i < self_energy_prefactor.size() ; i++ )
            e_self += self_energy_prefactor.at(i) * m2.at(i) * pow(invcutoff,2 * i + 1);
      return e_self;
    }
};
#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("[CoulombGalore] field-, force-, and energy-functions") {
    using doctest::Approx;
    double cutoff = 20.0;  // cutoff distance
    double zA = 2.0; // charge
    double zB = 2.0; // charge
    Point muA = {3, 0, 0}; // dipole
    Point muB = {3, 0, 0}; // dipole
    Point r = {11, 0, 0};  // distance vector

    PairPotential<plain> pot(cutoff);

    CHECK(pot.ion_field(zA, r) == Approx({0.016528925619835,0,0}));
    CHECK(pot.dipole_field(muA, r) == Approx({0,0,0})); // not implemented yet

    CHECK(pot.ion_ion_force(zA, zB, r) == Approx({0,0,0}));
    CHECK(pot.ion_dipole_force(zA, muB, r) == Approx({0,0,0}));
    CHECK(pot.dipole_dipole_force(muA, muB, r) == Approx({0,0,0}));

    CHECK(pot.ion_ion_energy(zA, zB, r) == Approx(0.363636363636364));
    CHECK(pot.ion_dipole_energy(zA, muB, r) == Approx(0));
    CHECK(pot.dipole_dipole_energy(muA, muB, r) == Approx(0));
}
#endif

// -------------- Plain ---------------

/**
 * @brief No truncation scheme
 */
struct Plain : public SchemeBase {
    inline Plain() : SchemeBase(TruncationScheme::plain, infty){};
    inline double splitting_function(double q) const override { return 1.0; };
    inline double calc_dielectric(double M2V) const override { return (2 * M2V + 1) / (1 - M2V); }
#ifdef NLOHMANN_JSON_HPP
  private:
    inline void _to_json(nlohmann::json &) const override {}
    inline void _from_json(const nlohmann::json &) override {}
#endif
};

#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("[CoulombGalore] plain") {
    using doctest::Approx;
    double zz = 2.0 * 2.0; // charge product
    Point r = {10, 0, 0};  // distance vector

    PairPotential<Plain> pot;
    CHECK(pot.splitting_function(0.5) == Approx(1.0));
    CHECK(pot.ion_ion_energy(zz, r.norm()) == Approx(zz / r.norm()));
}
#endif

// -------------- qPotential ---------------

/**
 * @brief qPotential scheme
 */
struct qPotential : public SchemeBase {
    double order; //!< Number of moment to cancel

    /**
     * @param cutoff distance cutoff
     * @param order number of moments to cancel
     */
    inline qPotential(double cutoff, double order) : SchemeBase(TruncationScheme::qpotential, cutoff), order(order) {
        name = "qpotential";
        self_energy_prefactor = {-1.0, -1.0};
    }

    inline double splitting_function(double q) const override { return qPochhammerSymbol(q, 1, order); }
    inline double calc_dielectric(double M2V) const override { return 1 + 3 * M2V; }

#ifdef NLOHMANN_JSON_HPP
  private:
    inline void _from_json(const nlohmann::json &j) override { order = j.at("order").get<double>(); }
    inline void _to_json(nlohmann::json &j) const override { j = {{"order", order}}; }
#endif
};

#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("[CoulombGalore] qPotential") {
    using doctest::Approx;
    double cutoff = 18.0;  // cutoff distance
    double zz = 2.0 * 2.0; // charge product
    Point r = {10, 0, 0};  // distance vector

    PairPotential<qPotential> pot(cutoff, 3);
    CHECK(pot.splitting_function(0.5) == Approx(0.328125));
    CHECK(pot.ion_ion_energy(zz, cutoff) == Approx(0));
    CHECK(pot.ion_ion_energy(zz, r.norm()) == Approx(0.1018333173));
}
#endif

// -------------- Poisson ---------------

/**
 * @brief Poisson approximation
 */
struct Poisson : public SchemeBase {
    unsigned int C;
    unsigned int D;

    inline Poisson(double cutoff, unsigned int C, unsigned int D)
        : SchemeBase(TruncationScheme::poisson, cutoff), C(C), D(D) {
        if ((C < 1) or (D < 1))
            throw std::runtime_error("`C` and `D` must be larger than zero");
        self_energy_prefactor = -double(C + D) / double(C);
    }
    inline double splitting_function(double q) const override {
        double tmp = 0;
        for (unsigned int c = 0; c < C; c++)
            tmp += double(factorial(D - 1 + c)) / double(factorial(D - 1)) / double(factorial(c)) * double(C - c) /
                   double(C) * std::pow(q, c);
        return std::pow(1 - q, D + 1) * tmp;
    }

    inline double calc_dielectric(double M2V) const override { return 1 + 3 * M2V; }

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
TEST_CASE("[CoulombGalore] Poisson") {}
#endif

// -------------- Fanourgakis ---------------

/**
 * @brief Fanourgakis scheme
 */
struct Fanourgakis : public SchemeBase {
    /**
     * @param cutoff distance cutoff
     */
    inline Fanourgakis(double cutoff) : SchemeBase(TruncationScheme::fanourgakis, cutoff) {
        name = "Fanourgakis";
        self_energy_prefactor = {-1.0, -1.0};
    }

    inline double splitting_function(double q) const override { return pow(1.0 - q,4.0) * ( 1.0 + 2.25 * q + 3.0 * q*q + 2.5 * q*q*q ); }
    inline double splitting_function_derivative(double q) const override { return ( -1.75 + 26.25 * pow(q,4.0) - 42.0 * pow(q,5.0) + 17.5 * pow(q,6.0) ); }
    inline double splitting_function_second_derivative(double q) const override { return 105.0 * pow(q,3.0) * pow(q - 1.0, 2); };
    inline double calc_dielectric(double M2V) const override { return 1 + 3 * M2V; }
};

#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("[CoulombGalore] Fanourgakis") {
    using doctest::Approx;
    double cutoff = 18.0;  // cutoff distance
    double zz = 2.0 * 2.0; // charge product
    Point r = {10, 0, 0};  // distance vector

    PairPotential<Fanourgakis> pot(cutoff);
    CHECK(pot.splitting_function(0.5) == Approx(0.1992187500));
    CHECK(pot.splitting_function_derivative(0.5) == Approx(-1.1484375));
    CHECK(pot.splitting_function_second_derivative(0.5) == Approx(3.28125));
    CHECK(pot.ion_ion_energy(zz, cutoff) == Approx(0));
    CHECK(pot.ion_ion_energy(zz, 0.0) == Approx(1));
    CHECK(pot.ion_ion_energy(zz, r.norm()) == Approx(0.1406456952));
}
#endif


} // namespace CoulombGalore
