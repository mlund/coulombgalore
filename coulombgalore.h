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
inline double qPochhammerSymbol(double q, int l = 0, int P = 300) {
    double value = 1.0;
    double qln = std::pow(q, l+1); // a * q^{n-1} = q^{l+n}
    for (int n = 1; n < P + 1; n++) {
        value *= ( 1.0 - qln );
        qln *= q;
    }
    return value;
}

inline double qPochhammerSymbolDerivative(double q, int l = 0, int P = 300) {
    double value = 0.0;
    double qln = std::pow(q, l+1); // a * q^{n-1} = q^{l+n}
    for (int n = 1; n < P + 1; n++) {
        value -= ( l + n ) * qln / ( 1.0 - qln );
        qln *= q;
    }
    return value / q * qPochhammerSymbol(q,l,P);
}

/**
 * @warning Does not work!
 */
inline double qPochhammerSymbolSecondDerivative(double q, int l = 0, int P = 300) {
    double value = 0.0;
    double qln = std::pow(q, l+1); // a * q^{n-1} = q^{l+n}
    for (int n = 1; n < P + 1; n++) {
        value -= ( qln + l + n - 1.0 ) * ( l + n ) * qln / ( 1.0 - qln ) / ( 1.0 - qln );
        qln *= q;
    }
    value /= ( q * q );
    double qPS = qPochhammerSymbol(q,l,P);
    double dqPS = qPochhammerSymbolDerivative(q,l,P);
    return ( value * qPS + dqPS * dqPS / qPS );
}

#ifdef DOCTEST_LIBRARY_INCLUDED
TEST_CASE("qPochhammerSymbol") {
    double q = 0.5;
    CHECK(qPochhammerSymbol(q, 0, 0) == 1);
    CHECK(qPochhammerSymbol(0, 0, 1) == 1);
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
    std::array<double,2> self_energy_prefactor; //!< Prefactor for self-energies
    inline SchemeBase(TruncationScheme scheme, double cutoff) : scheme(scheme), cutoff(cutoff) {}

    /**
     * @brief Splitting function
     * @param q q=r/Rcutoff
     * @todo How should this be expanded to higher order moments?
     */
    virtual double short_range_function(double) const = 0;

    virtual double short_range_function_derivative(double q, double dh=1e-9) {
        return ( short_range_function(q + dh) - short_range_function(q - dh) ) / ( 2 * dh );
    }

    virtual double short_range_function_second_derivative(double q, double dh=1e-9) {
        return ( short_range_function_derivative(q + dh , dh ) - short_range_function_derivative(q - dh , dh ) ) / ( 2 * dh );
    }

    virtual double short_range_function_third_derivative(double q, double dh=1e-9) {
        return ( short_range_function_second_derivative(q + dh , dh ) - short_range_function_second_derivative(q - dh , dh ) ) / ( 2 * dh );
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
    using Tscheme::short_range_function;
    using Tscheme::short_range_function_derivative;
    using Tscheme::short_range_function_second_derivative;
    using Tscheme::short_range_function_third_derivative;

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
        if ( r < cutoff ) {
            double q = r * invcutoff;
            return z / r * short_range_function(q);
        } else {
            return 0.0;
        }
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
            return mu.dot(r) / r2 / r1  * ( short_range_function(q) - q * short_range_function_derivative(q) );
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
            return z * r / r2 / r1 * ( short_range_function(q) - q * short_range_function_derivative(q) );
        } else {
            return {0,0,0};
        }
    }

    /**
     * @brief field from dipole
     * @returns field in electrostatic units ( why not Hartree atomic units? )
     * @param mu dipole
     * @param r distance-vector
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
            double second_derivative_scaled = q * q / 3.0 * short_range_function_second_derivative(q);
            Point field = ( 3.0 * mu.dot(r) * r / r2 - mu ) / r2 / r1 * ( short_range_function(q) - q * short_range_function_derivative(q) + second_derivative_scaled );
            field += mu / r2 / r1 * second_derivative_scaled;
            return field;
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
        return zB * ion_potential(zA,r);
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
        // Both expressions below gives same answer. Keep for possible optimization in future.
        //return -mu.dot(ion_field(z,r)); // field from charge interacting with dipole
        return z * dipole_potential(mu,-r); // potential of dipole interacting with charge
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
     * @brief ion-ion interaction force
     * @returns interaction force in electrostatic units ( why not Hartree atomic units? )
     * @param zA charge
     * @param zB charge
     * @param r distance-vector between charges
     */
    inline Point ion_ion_force(double zA, double zB, Point r) {
        return zB * ion_field(zA,r);
    }

    /**
     * @brief ion-dipole interaction force
     * @returns interaction force in electrostatic units ( why not Hartree atomic units? )
     * @param z charge
     * @param mu dipole moment
     * @param r distance-vector between dipole and charge, mu - z
     */
    inline Point ion_dipole_force(double z, Point mu, Point r) {
        return z * dipole_field(mu,r);
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
            Point force = 3.0 * ( ( 5.0 * muAdotRh*muBdotRh - muA.dot(muB) ) * rh - muBdotRh * muA - muAdotRh * muB  ) / r4;
            double second_derivative = short_range_function_second_derivative(q);
            force *= ( short_range_function(q) - q * short_range_function_derivative(q) + q * q / 3.0 * second_derivative );
            force += muAdotRh * muBdotRh * rh / r4 *( second_derivative - q * short_range_function_third_derivative(q) ) * q * q;
            return force;
        } else {
            return {0,0,0};
        }
    }

    /**
     * @brief torque exerted on dipole
     * @returns torque on dipole in electrostatic units ( why not Hartree atomic units? )
     * @param muA dipole moment
     * @param E field
     */
    inline Point dipole_torque(Point mu, Point E) {
        return mu.cross(E);
    }

    /**
     * @brief self-energy for all type of interactions
     * @param zz charge product
     * @returns self energy in electrostatic units ( why not Hartree atomic units? )
     * @param mumu product between dipole moment scalars
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
    inline Plain() : SchemeBase(TruncationScheme::plain, infty){
        name = "plain";
	doi = "Premier mémoire sur l’électricité et le magnétisme by Charles-Augustin de Coulomb"; // :P
        self_energy_prefactor = {0.0, 0.0};
    };
    inline double short_range_function(double q) const override { return 1.0; };
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
    CHECK(pot.ion_potential(zA, cutoff + 1.0) == Approx(0.06666666667 ));
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
    double d = 1e-5; // small distance

    Point muA_r1 = muA / muA.norm() * d;     // a small distance from the origin
    Point muA_r2 = - muA / muA.norm() * d;   // a small distance from the origin

    Point muB_r1 = r + muB / muB.norm() * d; // a small distance from 'r'
    Point muB_r2 = r - muB / muB.norm() * d; // a small distance from 'r'

    double muA_z1 =  muA.norm() / ( 2.0 * d ); // charge 1 of approximative dipole A
    double muA_z2 = -muA.norm() / ( 2.0 * d ); // charge 2 of approximative dipole A

    double muB_z1 =  muB.norm() / ( 2.0 * d ); // charge 1 of approximative dipole B
    double muB_z2 = -muB.norm() / ( 2.0 * d ); // charge 2 of approximative dipole B

    Point muA_approx = muA_r1 * muA_z1 + muA_r2 * muA_z2;
    Point muB_approx = muB_r1 * muB_z1 + muB_r2 * muB_z2;

    CHECK(muA[0] == Approx(muA_approx[0]));
    CHECK(muA[1] == Approx(muA_approx[1]));
    CHECK(muA[2] == Approx(muA_approx[2]));

    CHECK(muB[0] == Approx(muB_approx[0]));
    CHECK(muB[1] == Approx(muB_approx[1]));
    CHECK(muB[2] == Approx(muB_approx[2]));

    Point F_ionion_11 = pot.ion_ion_force(muA_z1, muB_z1, muA_r1 - muB_r1);
    Point F_ionion_12 = pot.ion_ion_force(muA_z1, muB_z2, muA_r1 - muB_r2);
    Point F_ionion_21 = pot.ion_ion_force(muA_z2, muB_z1, muA_r2 - muB_r1);
    Point F_ionion_22 = pot.ion_ion_force(muA_z2, muB_z2, muA_r2 - muB_r2);

    Point F_dipoledipole_approx = F_ionion_11 + F_ionion_12 + F_ionion_21 + F_ionion_22;

    CHECK(F_dipoledipole[0] == Approx(F_dipoledipole_approx[0]));
    CHECK(F_dipoledipole[1] == Approx(F_dipoledipole_approx[1]));
    CHECK(F_dipoledipole[2] == Approx(F_dipoledipole_approx[2]));
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
    }

    inline double short_range_function(double q) const override { return qPochhammerSymbol(q, 0, order); }
    inline double short_range_function_derivative(double q) const { return qPochhammerSymbolDerivative(q, 0, order); }
    inline double short_range_function_second_derivative(double q) const { return qPochhammerSymbolSecondDerivative(q, 0, order); }
    inline double calc_dielectric(double M2V) const override { return 1 + 3 * M2V; }

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
    double zA = 2.0; // charge
    double zB = 3.0; // charge
    Point muA = {19, 7, 11};  // dipole moment
    Point muB = {13, 17, 5};  // dipole moment
    Point r = {23, 0, 0};  // distance vector
    Point rh = {1, 0, 0};  // normalized distance vector

    PairPotential<qPotential> pot(cutoff,4);

    // Test short-ranged function
    CHECK(pot.short_range_function(0.5) == Approx(0.3076171875));
    CHECK(pot.short_range_function_derivative(0.5) == Approx(-1.453125));
}

#endif

// -------------- Poisson ---------------

/**
 * @brief Poisson approximation
 * @note By using the parameters 'C=4' and 'D=3' this equals the 'Fanourgakis' approach
 */
struct Poisson : public SchemeBase {
    unsigned int C;
    unsigned int D;

    inline Poisson(double cutoff, unsigned int C, unsigned int D)
        : SchemeBase(TruncationScheme::poisson, cutoff), C(C), D(D) {
        if ((C < 1) or (D < 1))
            throw std::runtime_error("`C` and `D` must be larger than zero");
        name = "poisson";
        doi = "10.1088/1367-2630/ab1ec1";
        double a1 = -double(C + D) / double(C);
        self_energy_prefactor = {a1, a1};
    }
    inline double short_range_function(double q) const override {
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
    }

    inline double short_range_function(double q) const override { return pow(1.0 - q,4.0) * ( 1.0 + 2.25 * q + 3.0 * q*q + 2.5 * q*q*q ); }
    inline double short_range_function_derivative(double q) const { return ( -1.75 + 26.25 * pow(q,4.0) - 42.0 * pow(q,5.0) + 17.5 * pow(q,6.0) ); }
    inline double short_range_function_second_derivative(double q) const { return 105.0 * pow(q,3.0) * pow(q - 1.0, 2); };
    inline double short_range_function_third_derivative(double q) const { return 525.0 * pow(q,2.0) * (q - 0.6) * ( q - 1.0); };
    inline double calc_dielectric(double M2V) const override { return 1 + 3 * M2V; }

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
    double zA = 2.0; // charge
    double zB = 3.0; // charge
    Point muA = {19, 7, 11};  // dipole moment
    Point muB = {13, 17, 5};  // dipole moment
    Point r = {23, 0, 0};  // distance vector
    Point rh = {1, 0, 0};  // normalized distance vector

    PairPotential<Fanourgakis> pot(cutoff);

    // Test short-ranged function
    CHECK(pot.short_range_function(0.5) == Approx(0.1992187500));
    CHECK(pot.short_range_function_derivative(0.5) == Approx(-1.1484375));
    CHECK(pot.short_range_function_second_derivative(0.5) == Approx(3.28125));
    CHECK(pot.short_range_function_third_derivative(0.5) == Approx(6.5625));

    // Test potentials
    CHECK(pot.ion_potential(zA, cutoff) == Approx(0.0));
    CHECK(pot.ion_potential(zA, r.norm()) == Approx(0.0009430652121));

    CHECK(pot.dipole_potential(muA, cutoff * rh) == Approx(0.0));
    CHECK(pot.dipole_potential(muA, r) == Approx(0.005750206554));

    // Test fields
    CHECK(pot.ion_field(zA, cutoff * rh).norm() == Approx(0.0));
    Point E_ion = pot.ion_field(zA, r);
    CHECK(E_ion[0] == Approx(0.0006052849004));
    CHECK(E_ion.norm() == Approx(0.0006052849004));

    CHECK(pot.dipole_field(muA, cutoff * rh).norm() == Approx(0.0));
    Point E_dipole = pot.dipole_field(muA, r);
    CHECK(E_dipole[0] == Approx(0.002702513754));
    CHECK(E_dipole[1] == Approx(-0.00009210857180));
    CHECK(E_dipole[2] == Approx(-0.0001447420414));

    // Test energies
    CHECK(pot.ion_ion_energy(zA, zB, cutoff) == Approx(0.0));
    CHECK(pot.ion_ion_energy(zA, zB, r.norm()) == Approx(0.002829195636));

    CHECK(pot.ion_dipole_energy(zA, muB, cutoff * rh) == Approx(0.0));
    CHECK(pot.ion_dipole_energy(zA, muB, r) == Approx(-0.007868703705));

    CHECK(pot.dipole_dipole_energy(muA, muB, cutoff * rh) == Approx(0.0));
    CHECK(pot.dipole_dipole_energy(muA, muB, r) == Approx(-0.03284312288));

    // Test forces
    CHECK(pot.ion_ion_force(zA, zB, cutoff * rh).norm() == Approx(0.0));
    Point F_ionion = pot.ion_ion_force(zA, zB, r);
    CHECK(F_ionion[0] == Approx(0.001815854701));
    CHECK(F_ionion.norm() == Approx(0.001815854701));

    CHECK(pot.ion_dipole_force(zB, muA, cutoff * rh).norm() == Approx(0.0));
    Point F_iondipole = pot.ion_dipole_force(zB, muA, r);
    CHECK(F_iondipole[0] == Approx(0.008107541263));
    CHECK(F_iondipole[1] == Approx(-0.0002763257154));
    CHECK(F_iondipole[2] == Approx(-0.0004342261242));

    CHECK(pot.dipole_dipole_force(muA, muB, cutoff * rh).norm() == Approx(0.0));
    Point F_dipoledipole = pot.dipole_dipole_force(muA, muB, r);
    CHECK(F_dipoledipole[0] == Approx(0.009216400961));
    CHECK(F_dipoledipole[1] == Approx(-0.002797126801));
    CHECK(F_dipoledipole[2] == Approx(-0.001608010094));

    // Approximate dipoles by two charges respectively and compare to point-dipoles
    double d = 1e-5; // small distance

    Point muA_r1 = muA / muA.norm() * d;     // a small distance from the origin
    Point muA_r2 = - muA / muA.norm() * d;   // a small distance from the origin

    Point muB_r1 = r + muB / muB.norm() * d; // a small distance from 'r'
    Point muB_r2 = r - muB / muB.norm() * d; // a small distance from 'r'

    double muA_z1 =  muA.norm() / ( 2.0 * d ); // charge 1 of approximative dipole A
    double muA_z2 = -muA.norm() / ( 2.0 * d ); // charge 2 of approximative dipole A

    double muB_z1 =  muB.norm() / ( 2.0 * d ); // charge 1 of approximative dipole B
    double muB_z2 = -muB.norm() / ( 2.0 * d ); // charge 2 of approximative dipole B

    Point muA_approx = muA_r1 * muA_z1 + muA_r2 * muA_z2;
    Point muB_approx = muB_r1 * muB_z1 + muB_r2 * muB_z2;

    CHECK(muA[0] == Approx(muA_approx[0]));
    CHECK(muA[1] == Approx(muA_approx[1]));
    CHECK(muA[2] == Approx(muA_approx[2]));

    CHECK(muB[0] == Approx(muB_approx[0]));
    CHECK(muB[1] == Approx(muB_approx[1]));
    CHECK(muB[2] == Approx(muB_approx[2]));

    Point F_ionion_11 = pot.ion_ion_force(muA_z1, muB_z1, muA_r1 - muB_r1);
    Point F_ionion_12 = pot.ion_ion_force(muA_z1, muB_z2, muA_r1 - muB_r2);
    Point F_ionion_21 = pot.ion_ion_force(muA_z2, muB_z1, muA_r2 - muB_r1);
    Point F_ionion_22 = pot.ion_ion_force(muA_z2, muB_z2, muA_r2 - muB_r2);

    Point F_dipoledipole_approx = F_ionion_11 + F_ionion_12 + F_ionion_21 + F_ionion_22;

    CHECK(F_dipoledipole[0] == Approx(F_dipoledipole_approx[0]));
    CHECK(F_dipoledipole[1] == Approx(F_dipoledipole_approx[1]));
    CHECK(F_dipoledipole[2] == Approx(F_dipoledipole_approx[2]));
}
#endif


} // namespace CoulombGalore
