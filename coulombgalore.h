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
    std::vector<double> self_energy_prefactor; //!< Prefactor for self-energies
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

    template <class... Args> PairPotential(Args &&... args) : Tscheme(args...) {
      invcutoff = 1.0 / cutoff;
      cutoff2 = cutoff*cutoff;
    }

    /**
     * @brief ion-ion interaction energy
     * @returns interaction energy in electrostatic units ( why not Hartree atomic units? )
     * @param zz charge product
     * @param r charge separation
     */
    inline double ion_ion_energy(double zz, double r) {
        if ( r < cutoff )
            return zz / r * splitting_function(r * invcutoff);
        else
            return 0.0;
    }

    /**
     * @brief field from ion
     * @returns field in electrostatic units ( why not Hartree atomic units? )
     * @param z charge
     * @param r distance-vector to charge
     */
    inline Point ion_field(double z, Point r) {
        double r2 = r.squaredNorm();
        if ( r2 < cutoff2 )
            return z / r2 * ( splitting_function(r * invcutoff) - r * invcutoff * splitting_function_derivative(r * invcutoff) ) * ( r / std::sqrt(r2) );
        else
            return {0,0,0};
    }

    /**
     * @brief ion-ion interaction force
     * @returns interaction force in electrostatic units ( why not Hartree atomic units? )
     * @param zz charge product
     * @param r distance-vector between charges
     */
    inline Point ion_ion_force(double zz, Point r) {
        return ion_field(zz,r);
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
        return ion_field(z,r).dot(mu);
    }

    /**
     * @brief ion-dipole interaction force
     * @returns interaction force in electrostatic units ( why not Hartree atomic units? )
     * @param z charge
     * @param mu dipole moment
     * @param r distance-vector between dipole and charge, mu - z
     */
    inline Point ion_dipole_force(double z, Point mu, Point r) {
        double r2 = r.squaredNorm();
        if ( r2 < cutoff2 ) {
	    double r1 = std::sqrt(r2);
            double q = r1 * invcutoff;
	    double b = splitting_function(q) - q * splitting_function_derivative(q);
            double a = b + q * q / 3.0 * splitting_function_second_derivative(q);
            return z * ( 3.0 * mu.dot(r) * r / r2 * a - mu * b ) / r2 / r1;
        } else {
            return {0,0,0};
        }
    }

    /**
     * @brief dipole-dipole interaction energy
     * @returns interaction energy in electrostatic units ( why not Hartree atomic units? )
     * @param muA dipole moment of particle A
     * @param muB dipole moment of particle B
     * @param r distance-vector between dipoles
     */
    inline double dipole_dipole_energy(Point muA, Point muB, Point r) {
        double r2 = r.squaredNorm();
        if ( r2 < cutoff2 ) {
            double r1 = std::sqrt(r2);
            double q = r1 * invcutoff;
            double b = q * q / 3.0 * splitting_function_second_derivative(q);
            double a = splitting_function(q) - q * splitting_function_derivative(q) + b;

            double dotproduct = muA.dot(muB);
            double T = (3 * muA.dot(r) * muB.dot(r) / r2 - dotproduct) * a + dotproduct * b;
            return - T / r2 / r1;
        } else {
            return 0.0;
        }
    }
    
    /**
     * @brief field from dipole
     * @returns field in electrostatic units ( why not Hartree atomic units? )
     * @param mu dipole
     * @param r distance-vector to dipole
     * @note not finished
     */
    inline Point dipole_field(Point mu, Point r) {
        double r2 = r.squaredNorm();
        if ( r2 < cutoff2 ) {
            return {0,0,0};
        } else {
            return {0,0,0};
        }
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
            return {0,0,0};
        } else {
            return {0,0,0};
        }
    }

    /**
     * @brief self-energy for all type of interactions
     * @param zz charge product
     * @returns self energy in electrostatic units ( why not Hartree atomic units? )
     * @param mumu product between dipole moment scalars
     */
    inline double self_energy(std::vector<double> m2) const {
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
