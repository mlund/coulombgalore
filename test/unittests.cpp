#undef DOCTEST_CONFIG_DISABLE

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS

#include <doctest/doctest.h>
#include <nlohmann/json.hpp>
#include "coulombgalore.h"

using namespace CoulombGalore;

TEST_CASE("powi") {
    using doctest::Approx;
    double x = 3.1;
    CHECK(powi(-x, -1) == Approx(std::pow(-x, -1)));
    CHECK(powi(x, -1) == Approx(std::pow(x, -1)));
    CHECK(powi(x, 0) == Approx(1));
    CHECK(powi(x, 1) == Approx(x));
    CHECK(powi(x, 2) == Approx(x * x));
    CHECK(powi(x, 4) == Approx(x * x * x * x));
}

TEST_CASE("Factorial") {
    CHECK(factorial(0) == 1);
    CHECK(factorial(1) == 1);
    CHECK(factorial(2) == 2);
    CHECK(factorial(3) == 6);
    CHECK(factorial(10) == 3628800);
}
TEST_CASE("Binomial") {
    CHECK(binomial(3, 2) == 3);
    CHECK(binomial(5, 2) == 10);
    CHECK(binomial(8, 3) == 56);
    CHECK(binomial(9, 7) == 36);
    CHECK(binomial(5, 0) == 1);
    CHECK(binomial(12, 1) == 12);
    CHECK(binomial(11, 11) == 1);
    CHECK(binomial(2, 0) == 1);
    CHECK(binomial(3, 1) == 3);
    CHECK(binomial(4, 2) == 6);
    CHECK(binomial(5, 3) == 10);
}
// numerical differentioation used for unittests, only
inline double diff1(std::function<double(double)> f, double x, double dx = 1e-4) {
    return (f(x + dx) - f(x - dx)) / (2 * dx);
}

inline double diff2(std::function<double(double)> f, double x, double dx = 1e-4) {
    return (diff1(f, x + dx, dx) - diff1(f, x - dx, dx)) / (2 * dx);
}

inline double diff3(std::function<double(double)> f, double x, double dx = 1e-4) {
    return (diff2(f, x + dx, dx) - diff2(f, x - dx, dx)) / (2 * dx);
}

// Compare differentiation with numerical differentiation.
template <class Potential> void testDerivatives(Potential &pot, double q) {
    using doctest::Approx;
    auto s = std::bind(&Potential::short_range_function, pot, std::placeholders::_1);
    CHECK(s(q) == Approx(pot.short_range_function(q)));
    CHECK(diff1(s, q) == Approx(pot.short_range_function_derivative(q)));
    CHECK(diff2(s, q) == Approx(pot.short_range_function_second_derivative(q)));
    CHECK(diff3(s, q) == Approx(pot.short_range_function_third_derivative(q)));
}

TEST_CASE("qPochhammerSymbol") {
    using doctest::Approx;
    CHECK(qPochhammerSymbol(0.5, 0, 0) == 1);
    CHECK(qPochhammerSymbol(0, 0, 1) == 1);
    CHECK(qPochhammerSymbol(1, 0, 1) == 0);
    CHECK(qPochhammerSymbol(1, 1, 2) == 0);
    CHECK(qPochhammerSymbol(0.75, 0, 2) == Approx(0.109375));
    CHECK(qPochhammerSymbol(2.0 / 3.0, 2, 5) == Approx(0.4211104676));
    CHECK(qPochhammerSymbol(0.125, 1, 1) == Approx(0.984375));
    CHECK(qPochhammerSymbolDerivative(0.75, 0, 2) == Approx(-0.8125));
    CHECK(qPochhammerSymbolDerivative(2.0 / 3.0, 2, 5) == Approx(-2.538458169));
    CHECK(qPochhammerSymbolDerivative(0.125, 1, 1) == Approx(-0.25));
    CHECK(qPochhammerSymbolSecondDerivative(0.75, 0, 2) == Approx(2.5));
    CHECK(qPochhammerSymbolSecondDerivative(2.0 / 3.0, 2, 5) == Approx(-1.444601767));
    CHECK(qPochhammerSymbolSecondDerivative(0.125, 1, 1) == Approx(-2.0));
    CHECK(qPochhammerSymbolThirdDerivative(0.75, 0, 2) == Approx(6.0));
    CHECK(qPochhammerSymbolThirdDerivative(2.0 / 3.0, 2, 5) == Approx(92.48631425));
    CHECK(qPochhammerSymbolThirdDerivative(0.125, 1, 1) == Approx(0.0));
    CHECK(qPochhammerSymbolThirdDerivative(0.4, 3, 7) == Approx(-32.80472205));
}

TEST_CASE("[CoulombGalore] Andrea") {
    using doctest::Approx;
    using namespace Tabulate;

    auto f = [](double x) { return 0.5 * x * std::sin(x) + 2; };
    Andrea<double> spline;
    spline.setTolerance(2e-6, 1e-4); // ftol carries no meaning
    auto d = spline.generate(f, 0, 10);

    CHECK(spline.eval(d, 1e-9) == Approx(f(1e-9)));
    CHECK(spline.eval(d, 5) == Approx(f(5)));
    CHECK(spline.eval(d, 10) == Approx(f(10)));

    // Check if numerical derivation of *splined* function
    // matches the analytical solution in `evalDer()`.
    auto f_prime = [&](double x, double dx = 1e-10) {
        return (spline.eval(d, x + dx) - spline.eval(d, x - dx)) / (2 * dx);
    };
    double x = 1e-9;
    CHECK(spline.evalDer(d, x) == Approx(f_prime(x)));
    x = 1;
    CHECK(spline.evalDer(d, x) == Approx(f_prime(x)));
    x = 5;
    CHECK(spline.evalDer(d, x) == Approx(f_prime(x)));

    // Check if analytical spline derivative matches
    // derivative of original function
    auto f_prime_exact = [&](double x, double dx = 1e-10) { return (f(x + dx) - f(x - dx)) / (2 * dx); };
    x = 1e-9;
    CHECK(spline.evalDer(d, x) == Approx(f_prime_exact(x)));
    x = 1;
    CHECK(spline.evalDer(d, x) == Approx(f_prime_exact(x)));
    x = 5;
    CHECK(spline.evalDer(d, x) == Approx(f_prime_exact(x)));
}

TEST_CASE("[CoulombGalore] plain") {
    using doctest::Approx;
    double cutoff = 29.0;   // cutoff distance
    double zA = 2.0;        // charge
    double zB = 3.0;        // charge
    vec3 muA = {19, 7, 11}; // dipole moment
    vec3 muB = {13, 17, 5}; // dipole moment
    mat33 quadA, quadB;     // quadrupole moment
    quadA << 3, 7, 8, 5, 9, 6, 2, 1, 4;
    quadB << 0, 0, 0, 0, 0, 0, 0, 0, 0;
    vec3 r = {23, 0, 0};    // distance vector
    vec3 rq = {5.75*std::sqrt(6.0), 5.75*std::sqrt(2.0), 11.5*std::sqrt(2.0)};    // distance vector for quadrupole check
    vec3 rh = {1, 0, 0};    // normalized distance vector
    Plain pot;

    // Test self energy
    CHECK(pot.self_energy({zA * zB, muA.norm() * muB.norm()}) == Approx(0.0));

    // Test short-ranged function
    CHECK(pot.short_range_function(0.5) == Approx(1.0));
    CHECK(pot.short_range_function_derivative(0.5) == Approx(0.0));
    CHECK(pot.short_range_function_second_derivative(0.5) == Approx(0.0));
    CHECK(pot.short_range_function_third_derivative(0.5) == Approx(0.0));

    testDerivatives(pot, 0.5); // Compare differentiation with numerical diff.

    // Test potentials
    CHECK(pot.ion_potential(zA, cutoff + 1.0) == Approx(0.06666666667));
    CHECK(pot.ion_potential(zA, r.norm()) == Approx(0.08695652174));
    CHECK(pot.dipole_potential(muA, (cutoff + 1.0) * rh) == Approx(0.02111111111));
    CHECK(pot.dipole_potential(muA, r) == Approx(0.03591682420));
    CHECK(pot.quadrupole_potential(quadA, rq) == Approx(0.00093632817));

    // Test fields
    CHECK(pot.ion_field(zA, (cutoff + 1.0) * rh).norm() == Approx(0.002222222222));
    vec3 E_ion = pot.ion_field(zA, r);
    CHECK(E_ion[0] == Approx(0.003780718336));
    CHECK(E_ion.norm() == Approx(0.003780718336));
    CHECK(pot.dipole_field(muA, (cutoff + 1.0) * rh).norm() == Approx(0.001487948846));
    vec3 E_dipole = pot.dipole_field(muA, r);
    CHECK(E_dipole[0] == Approx(0.003123202104));
    CHECK(E_dipole[1] == Approx(-0.0005753267034));
    CHECK(E_dipole[2] == Approx(-0.0009040848196));
    vec3 E_quadrupole = pot.quadrupole_field(quadA, r);
    CHECK(E_quadrupole[0] == Approx(-0.00003752130674));
    CHECK(E_quadrupole[1] == Approx(-0.00006432224013));
    CHECK(E_quadrupole[2] == Approx(-0.00005360186677));

    // Test energies
    CHECK(pot.ion_ion_energy(zA, zB, (cutoff + 1.0)) == Approx(0.2));
    CHECK(pot.ion_ion_energy(zA, zB, r.norm()) == Approx(0.2608695652));
    CHECK(pot.ion_dipole_energy(zA, muB, (cutoff + 1.0) * rh) == Approx(-0.02888888889));
    CHECK(pot.ion_dipole_energy(zA, muB, r) == Approx(-0.04914933837));
    CHECK(pot.dipole_dipole_energy(muA, muB, (cutoff + 1.0) * rh) == Approx(-0.01185185185));
    CHECK(pot.dipole_dipole_energy(muA, muB, r) == Approx(-0.02630064930));
    CHECK(pot.ion_quadrupole_energy(zB, quadA, rq) == Approx(0.002808984511));

    // Test forces
    CHECK(pot.ion_ion_force(zA, zB, (cutoff + 1.0) * rh).norm() == Approx(0.006666666667));
    vec3 F_ionion = pot.ion_ion_force(zA, zB, r);
    CHECK(F_ionion[0] == Approx(0.01134215501));
    CHECK(F_ionion.norm() == Approx(0.01134215501));
    CHECK(pot.ion_dipole_force(zB, muA, (cutoff + 1.0) * rh).norm() == Approx(0.004463846540));
    vec3 F_iondipole = pot.ion_dipole_force(zB, muA, r);
    CHECK(F_iondipole[0] == Approx(0.009369606312));
    CHECK(F_iondipole[1] == Approx(-0.001725980110));
    CHECK(F_iondipole[2] == Approx(-0.002712254459));
    CHECK(pot.dipole_dipole_force(muA, muB, (cutoff + 1.0) * rh).norm() == Approx(0.002129033733));
    vec3 F_dipoledipole = pot.dipole_dipole_force(muA, muB, r);
    CHECK(F_dipoledipole[0] == Approx(0.003430519474));
    CHECK(F_dipoledipole[1] == Approx(-0.004438234569));
    CHECK(F_dipoledipole[2] == Approx(-0.002551448858));

    // Approximate dipoles by two charges respectively and compare to point-dipoles
    double d = 1e-3;                          // small distance
    vec3 r_muA_1 = muA / muA.norm() * d;      // a small distance from dipole A ( the origin )
    vec3 r_muA_2 = -muA / muA.norm() * d;     // a small distance from dipole B ( the origin )
    vec3 r_muB_1 = r + muB / muB.norm() * d;  // a small distance from dipole B ( 'r' )
    vec3 r_muB_2 = r - muB / muB.norm() * d;  // a small distance from dipole B ( 'r' )
    double z_muA_1 = muA.norm() / (2.0 * d);  // charge 1 of approximative dipole A
    double z_muA_2 = -muA.norm() / (2.0 * d); // charge 2 of approximative dipole A
    double z_muB_1 = muB.norm() / (2.0 * d);  // charge 1 of approximative dipole B
    double z_muB_2 = -muB.norm() / (2.0 * d); // charge 2 of approximative dipole B
    vec3 muA_approx = r_muA_1 * z_muA_1 + r_muA_2 * z_muA_2;
    vec3 muB_approx = r_muB_1 * z_muB_1 + r_muB_2 * z_muB_2;
    vec3 r_z1r = r - r_muA_1; // distance from charge 1 of dipole A to 'r'
    vec3 r_z2r = r - r_muA_2; // distance from charge 2 of dipole A to 'r'

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
    vec3 fieldA = pot.dipole_field(muA, r);
    vec3 fieldA_1 = pot.ion_field(z_muA_1, r_z1r);
    vec3 fieldA_2 = pot.ion_field(z_muA_2, r_z2r);
    CHECK(fieldA[0] == Approx(fieldA_1[0] + fieldA_2[0]));
    CHECK(fieldA[1] == Approx(fieldA_1[1] + fieldA_2[1]));
    CHECK(fieldA[2] == Approx(fieldA_1[2] + fieldA_2[2]));

    // Check energies
    double EA = pot.ion_dipole_energy(zB, muA, -r);
    double EA_1 = pot.ion_ion_energy(zB, z_muA_1, r_z1r.norm());
    double EA_2 = pot.ion_ion_energy(zB, z_muA_2, r_z2r.norm());
    CHECK(EA == Approx(EA_1 + EA_2));

    // Check forces
    vec3 F_ionion_11 = pot.ion_ion_force(z_muA_1, z_muB_1, r_muA_1 - r_muB_1);
    vec3 F_ionion_12 = pot.ion_ion_force(z_muA_1, z_muB_2, r_muA_1 - r_muB_2);
    vec3 F_ionion_21 = pot.ion_ion_force(z_muA_2, z_muB_1, r_muA_2 - r_muB_1);
    vec3 F_ionion_22 = pot.ion_ion_force(z_muA_2, z_muB_2, r_muA_2 - r_muB_2);
    vec3 F_dipoledipole_approx = F_ionion_11 + F_ionion_12 + F_ionion_21 + F_ionion_22;
    CHECK(F_dipoledipole[0] == Approx(F_dipoledipole_approx[0]));
    CHECK(F_dipoledipole[1] == Approx(F_dipoledipole_approx[1]));
    CHECK(F_dipoledipole[2] == Approx(F_dipoledipole_approx[2]));

    // Approximate a quadrupole by four charges and compare to point-quadrupole
    d = 1e-4;               // small distance
    mat33 quad0;            // quadrupole moment
    quad0 << -2.38, -0.42, 2.08, -0.42, 0.00, 0.24, 2.08, 0.24, -1.60;
    vec3 r_quad0_1 = {0.5*d, 0.2*d, 1.0*d};    // distance vector to charge 1
    vec3 r_quad0_2 = {1.0*d, 0.5*d, 0.2*d};    // distance vector to charge 2
    vec3 r_quad0_3 = {0.3*d, 0.5*d, 0.6*d};    // distance vector to charge 3
    vec3 r_quad0_4 = {-0.9*d, 0.2*d, 1.8*d};    // distance vector to charge 4
    r_z1r = r - r_quad0_1; // distance from charge 1 of quadrupole A to 'r'
    r_z2r = r - r_quad0_2; // distance from charge 2 of quadrupole A to 'r'
    vec3 r_z3r = r - r_quad0_3; // distance from charge 1 of quadrupole A to 'r'
    vec3 r_z4r = r - r_quad0_4; // distance from charge 2 of quadrupole A to 'r'
    double z_quad0_1 = 1.0 / d/ d;
    double z_quad0_2 = -2.0 / d/ d;
    double z_quad0_3 = 2.0 / d/ d;
    double z_quad0_4 = -1.0 / d/ d;

    mat33 quad = r_quad0_1*r_quad0_1.transpose()*z_quad0_1 + r_quad0_2*r_quad0_2.transpose()*z_quad0_2 + r_quad0_3*r_quad0_3.transpose()*z_quad0_3 + r_quad0_4*r_quad0_4.transpose()*z_quad0_4;
    CHECK(quad(0,0) == Approx(quad0(0,0)));
    CHECK(quad(0,1) == Approx(quad0(0,1)));
    CHECK(quad(0,2) == Approx(quad0(0,2)));
    CHECK(quad(1,0) == Approx(quad0(1,0)));
    CHECK(quad(1,1) == Approx(quad0(1,1)));
    CHECK(quad(1,2) == Approx(quad0(1,2)));
    CHECK(quad(2,0) == Approx(quad0(2,0)));
    CHECK(quad(2,1) == Approx(quad0(2,1)));
    CHECK(quad(2,2) == Approx(quad0(2,2)));

    // Check potentials
    double pot0 = pot.quadrupole_potential(quad0, r);
    double pot0_1 = pot.ion_potential(z_quad0_1, r_z1r.norm());
    double pot0_2 = pot.ion_potential(z_quad0_2, r_z2r.norm());
    double pot0_3 = pot.ion_potential(z_quad0_3, r_z3r.norm());
    double pot0_4 = pot.ion_potential(z_quad0_4, r_z4r.norm());
    CHECK(pot0 == Approx(pot0_1 + pot0_2 + pot0_3 + pot0_4));

    // Check fields
    vec3 fie0 = pot.quadrupole_field(quad0, r);
    vec3 fie0_1 = pot.ion_field(z_quad0_1, r_z1r);
    vec3 fie0_2 = pot.ion_field(z_quad0_2, r_z2r);
    vec3 fie0_3 = pot.ion_field(z_quad0_3, r_z3r);
    vec3 fie0_4 = pot.ion_field(z_quad0_4, r_z4r);
    CHECK(fie0[0] == Approx(fie0_1[0] + fie0_2[0] + fie0_3[0] + fie0_4[0]));
    CHECK(fie0[1] == Approx(fie0_1[1] + fie0_2[1] + fie0_3[1] + fie0_4[1]));
    CHECK(fie0[2] == Approx(fie0_1[2] + fie0_2[2] + fie0_3[2] + fie0_4[2]));

    // Check Yukawa-interactions
    double debye_length = 23.0;
    Plain potY(debye_length);

    // Test potentials
    CHECK(potY.ion_potential(zA, cutoff + 1.0) == Approx(0.01808996296));
    CHECK(potY.ion_potential(zA, r.norm()) == Approx(0.03198951663));
    CHECK(potY.dipole_potential(muA, (cutoff + 1.0) * rh) == Approx(0.01320042949));
    CHECK(potY.dipole_potential(muA, r) == Approx(0.02642612243));

    // Test fields
    CHECK(potY.ion_field(zA, (cutoff + 1.0) * rh).norm() == Approx(0.001389518894));
    vec3 E_ion_Y = potY.ion_field(zA, r);
    CHECK(E_ion_Y[0] == Approx(0.002781697098));
    CHECK(E_ion_Y.norm() == Approx(0.002781697098));
    CHECK(potY.dipole_field(muA, (cutoff + 1.0) * rh).norm() == Approx(0.001242154748));
    vec3 E_dipole_Y = potY.dipole_field(muA, r);
    CHECK(E_dipole_Y[0] == Approx(0.002872404612));
    CHECK(E_dipole_Y[1] == Approx(-0.0004233017324));
    CHECK(E_dipole_Y[2] == Approx(-0.0006651884364));

    // Test forces
    CHECK(potY.dipole_dipole_force(muA, muB, (cutoff + 1.0) * rh).norm() == Approx(0.001859094075));
    vec3 F_dipoledipole_Y = potY.dipole_dipole_force(muA, muB, r);
    CHECK(F_dipoledipole_Y[0] == Approx(0.003594120919));
    CHECK(F_dipoledipole_Y[1] == Approx(-0.003809715590));
    CHECK(F_dipoledipole_Y[2] == Approx(-0.002190126354));
}

TEST_CASE("[CoulombGalore] Wolf") {
    using doctest::Approx;
    double cutoff = 29.0; // cutoff distance
    double alpha = 0.1;   // damping-parameter
    Wolf pot(cutoff, alpha);

    CHECK(pot.self_energy({4.0, 0.0}) == Approx(-0.2256786677));
    CHECK(pot.self_energy({0.0, 2.0}) == Approx(-0.0007522527778));

    // Test short-ranged function
    CHECK(pot.short_range_function(0.5) == Approx(0.04028442542));
    CHECK(pot.short_range_function_derivative(0.5) == Approx(-0.3997546829));
    CHECK(pot.short_range_function_second_derivative(0.5) == Approx(3.361591250));
    CHECK(pot.short_range_function_third_derivative(0.5) == Approx(-21.54779991));
    CHECK(pot.short_range_function(1.0) == Approx(0.0));
    testDerivatives(pot, 0.5); // Compare differentiation with numerical diff.
}

TEST_CASE("[CoulombGalore] Zahn") {
    using doctest::Approx;
    double cutoff = 29.0; // cutoff distance
    double alpha = 0.1;   // damping-parameter
    Zahn pot(cutoff, alpha);

    CHECK(pot.self_energy({4.0, 0.0}) == Approx(-0.2256227568));
    CHECK(pot.self_energy({0.0, 2.0}) == Approx(0.0));

    // Test short-ranged function
    CHECK(pot.short_range_function(0.5) == Approx(0.04049737684));
    CHECK(pot.short_range_function_derivative(0.5) == Approx(-0.3997135850));
    CHECK(pot.short_range_function_second_derivative(0.5) == Approx(3.360052030));
    CHECK(pot.short_range_function_third_derivative(0.5) == Approx(-21.54779991));
    CHECK(pot.short_range_function(1.0) == Approx(0.0000410979));
    testDerivatives(pot, 0.5); // Compare differentiation with numerical diff.
}

TEST_CASE("[CoulombGalore] Fennell") {
    using doctest::Approx;
    double cutoff = 29.0; // cutoff distance
    double alpha = 0.1;   // damping-parameter
    Fennell pot(cutoff, alpha);

    CHECK(pot.self_energy({4.0, 0.0}) == Approx(-0.2257317442));
    CHECK(pot.self_energy({0.0, 2.0}) == Approx(0.0));

    // Test short-ranged function
    CHECK(pot.short_range_function(0.5) == Approx(0.04009202308));
    CHECK(pot.short_range_function_derivative(0.5) == Approx(-0.3997546830));
    CHECK(pot.short_range_function_second_derivative(0.5) == Approx(3.363130468));
    CHECK(pot.short_range_function_third_derivative(0.5) == Approx(-21.54779991));
    CHECK(pot.short_range_function(1.0) == Approx(0.0));
    testDerivatives(pot, 0.5); // Compare differentiation with numerical diff.
}

TEST_CASE("[CoulombGalore] ZeroDipole") {
    using doctest::Approx;
    double cutoff = 29.0; // cutoff distance
    double alpha = 0.1;   // damping-parameter
    ZeroDipole pot(cutoff, alpha);

    CHECK(pot.self_energy({4.0, 0.0}) == Approx(-0.2257052059));
    CHECK(pot.self_energy({0.0, 2.0}) == Approx(-0.0007522843336));

    // Test short-ranged function
    CHECK(pot.short_range_function(0.5) == Approx(0.04014012381));
    CHECK(pot.short_range_function_derivative(0.5) == Approx(-0.3998508840));
    CHECK(pot.short_range_function_second_derivative(0.5) == Approx(3.362745664));
    CHECK(pot.short_range_function_third_derivative(0.5) == Approx(-21.54549108));
    CHECK(pot.short_range_function(1.0) == Approx(0.0));
    testDerivatives(pot, 0.5); // Compare differentiation with numerical diff.
}

TEST_CASE("[CoulombGalore] Reaction-field") {
    using doctest::Approx;
    double cutoff = 29.0; // cutoff distance
    double epsr = 1.0;    // diel. const. of sample
    double epsRF = 80.0;  // diel. const. of surrounding
    bool shifted = false;
    ReactionField pot(cutoff, epsRF, epsr, shifted);

    CHECK(pot.self_energy({4.0, 0.0}) == Approx(0.0));
    CHECK(pot.self_energy({0.0, 2.0}) == Approx(-0.00004023807698));

    // Test short-ranged function
    CHECK(pot.short_range_function(0.5) == Approx(1.061335404));
    CHECK(pot.short_range_function_derivative(0.5) == Approx(0.3680124224));
    CHECK(pot.short_range_function_second_derivative(0.5) == Approx(1.472049689));
    CHECK(pot.short_range_function_third_derivative(0.5) == Approx(2.944099379));
    CHECK(pot.short_range_function(1.0) == Approx(1.490683230));
    testDerivatives(pot, 0.5); // Compare differentiation with numerical diff.

    shifted = true;
    ReactionField potS(cutoff, epsRF, epsr, shifted);

    CHECK(potS.self_energy({4.0, 0.0}) == Approx(-0.1028057400));
    CHECK(potS.self_energy({0.0, 2.0}) == Approx(-0.00004023807698));

    // Test short-ranged function
    CHECK(potS.short_range_function(0.5) == Approx(0.3159937888));
    CHECK(potS.short_range_function_derivative(0.5) == Approx(-1.122670807));
    CHECK(potS.short_range_function_second_derivative(0.5) == Approx(1.472049689));
    CHECK(potS.short_range_function_third_derivative(0.5) == Approx(2.944099379));
    CHECK(potS.short_range_function(1.0) == Approx(0.0));
    // testDerivatives(potS, 0.5); // Compare differentiation with numerical diff.
}

TEST_CASE("[CoulombGalore] qPotential") {
    using doctest::Approx;
    double cutoff = 29.0; // cutoff distance
    int order = 4;        // number of higher order moments to cancel - 1
    qPotential pot(cutoff, order);

    CHECK(pot.self_energy({4.0, 0.0}) == Approx(-0.06896551724));
    CHECK(pot.self_energy({0.0, 2.0}) == Approx(-0.000041002091105));

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

    testDerivatives(pot, 0.5); // Compare differentiation with numerical diff.
}
TEST_CASE("[CoulombGalore] Fanourgakis") {
    using doctest::Approx;
    double cutoff = 29.0; // cutoff distance
    Fanourgakis pot(cutoff);

    CHECK(pot.self_energy({4.0, 0.0}) == Approx(-0.120689655172414));
    CHECK(pot.self_energy({0.0, 2.0}) == Approx(0.0));

    // Test short-ranged function
    CHECK(pot.short_range_function(0.5) == Approx(0.1992187500));
    CHECK(pot.short_range_function_derivative(0.5) == Approx(-1.1484375));
    CHECK(pot.short_range_function_second_derivative(0.5) == Approx(3.28125));
    CHECK(pot.short_range_function_third_derivative(0.5) == Approx(6.5625));

    testDerivatives(pot, 0.5); // Compare differentiation with numerical diff.

    // Possion with D=4, C=3 should equal Fanourgakis:
    CHECK(Poisson(cutoff, 4, 3).short_range_function(0.5) == Approx(pot.short_range_function(0.5)));
}

TEST_CASE("[CoulombGalore] Ewald (Gaussian) real-space") {
    using doctest::Approx;
    double cutoff = 29.0; // cutoff distance
    double alpha = 0.1;   // damping-parameter
    double eps_sur = infinity;
    Ewald pot(cutoff, alpha, eps_sur);

    CHECK(pot.self_energy({4.0, 0.0}) == Approx(-0.2256758334));
    CHECK(pot.self_energy({0.0, 2.0}) == Approx(-0.0007522527778));

    // Test short-ranged function
    CHECK(pot.short_range_function(0.5) == Approx(0.04030497436));
    CHECK(pot.short_range_function_derivative(0.5) == Approx(-0.399713585));
    CHECK(pot.short_range_function_second_derivative(0.5) == Approx(3.36159125));
    CHECK(pot.short_range_function_third_derivative(0.5) == Approx(-21.54779991));

    std::vector<double> charges = {1.0, -1.0, 2.0, 3.0};
    double volume = 2.0;
    CHECK(pot.neutralization_energy(charges, volume) == Approx(-1963.341583564));

    testDerivatives(pot, 0.5); // Compare differentiation with numerical diff.

    double debye_length = 23.0;
    Ewald potY(cutoff, alpha, eps_sur, debye_length);

    CHECK(potY.self_energy({4.0, 0.0}) == Approx(-0.1493013040));
    CHECK(potY.self_energy({0.0, 2.0}) == Approx(-0.0006704901976));

    // Test short-ranged function
    CHECK(potY.short_range_function(0.5) == Approx(0.07306333588));
    CHECK(potY.short_range_function_derivative(0.5) == Approx(-0.63444119));
    CHECK(potY.short_range_function_second_derivative(0.5) == Approx(4.423133599));
    CHECK(potY.short_range_function_third_derivative(0.5) == Approx(-19.85937171));

    CHECK(potY.neutralization_energy(charges, volume) == Approx(-1917.674014375));

    // Test against DOI: 10.1063/1.481216 and DOI: 10.1063/1.3599045
    std::vector<vec3> positions;
    std::vector<vec3> dipoles;
    positions.resize(2);
    charges.resize(2);
    dipoles.resize(2);
    vec3 zeros = {0.0, 0.0, 0.0};

    double L1 = 10.0;
    volume = L1*L1*L1;
    vec3 L = {L1, L1, L1};
    int nmax = 11;
    cutoff = L1 / 2.0;
    alpha = 8.0 / L1;
    vec3 rA = {-0.5, 0.0, 0.0};
    vec3 rB = {0.5, 0.0, 0.0};
    positions.at(0) = rA;
    positions.at(1) = rB;
    vec3 rAB = rA - rB;

    // -- two charges
    double qA = 1.0;
    double qB = -1.0;
    vec3 muA = {0.0, 0.0, 0.0};
    vec3 muB = {0.0, 0.0, 0.0};
    charges.at(0) = qA;
    charges.at(1) = qB;
    dipoles.at(0) = muA;
    dipoles.at(1) = muB;

    // ---- infinite diel. coeff. of surrounding
    eps_sur = infinity;
    Ewald pot_reci_inf(cutoff, alpha, eps_sur);

    
    double real_energy = pot_reci_inf.ion_ion_energy(qA, qB, rAB.norm());
    double self_energy = pot_reci_inf.self_energy({qA*qA + qB*qB, 0.0});
    double surface_energy = pot_reci_inf.surface_energy(positions, charges, dipoles, volume);
    double reciprocal_energy = pot_reci_inf.reciprocal_energy(positions,charges,dipoles,L,nmax);
    CHECK(real_energy == Approx(-0.2578990353));
    CHECK(self_energy == Approx(-0.9027033337));
    CHECK(surface_energy == Approx(0.0));
    CHECK(reciprocal_energy == Approx(0.1584768302));
    CHECK(real_energy+self_energy+surface_energy+reciprocal_energy == Approx(-1.0021255));

    EwaldData data;
    data.surface_dielectric_constant = eps_sur;
    data.cutoff = cutoff;
    data.reciprocal_cutoff = nmax;
    data.alpha = alpha;
    pot_reci_inf.kvectorUpdate(data, L);

    SUBCASE("EwaldData energy") {
        double reciprocal_energy = pot_reci_inf.reciprocal_energy(data, positions, charges, dipoles);
        CHECK(reciprocal_energy == Approx(0.1584768302));
    }

    vec3 real_force = pot_reci_inf.ion_ion_force(qA, qB, rAB);
    CHECK(real_force[0] == Approx(0.7338876643));
    CHECK(real_force[1] == Approx(0.0));
    CHECK(real_force[2] == Approx(0.0));

    vec3 surface_force = pot_reci_inf.surface_force(positions, charges, dipoles, volume, qA);
    CHECK(surface_force[0] == Approx(0.0));
    CHECK(surface_force[1] == Approx(0.0));
    CHECK(surface_force[2] == Approx(0.0));

    vec3 reciprocal_force = pot_reci_inf.reciprocal_force(positions, charges, dipoles, L, nmax, rA, qA, muA);
    CHECK(reciprocal_force[0] == Approx(0.2617988467));
    CHECK(reciprocal_force[1] == Approx(0.0));
    CHECK(reciprocal_force[2] == Approx(0.0));

    CHECK(real_force[0]+reciprocal_force[0]+surface_force[0] == Approx(0.9956865));

    vec3 real_field = pot_reci_inf.ion_field(qB, rAB);
    CHECK(real_field[0] == Approx(0.7338876643));
    CHECK(real_field[1] == Approx(0.0));
    CHECK(real_field[2] == Approx(0.0));

    vec3 reciprocal_field = pot_reci_inf.reciprocal_field(positions, charges, dipoles, L, nmax, rA);
    CHECK(reciprocal_field[0] == Approx(0.2617988467));
    CHECK(reciprocal_field[1] == Approx(0.0));
    CHECK(reciprocal_field[2] == Approx(0.0));

    SUBCASE("EwaldData field") {
        auto reciprocal_field = pot_reci_inf.reciprocal_field(data, positions, charges, dipoles, rA);
        CHECK(reciprocal_field[0] == Approx(0.2617988467));
        CHECK(reciprocal_field[1] == Approx(0.0));
        CHECK(reciprocal_field[2] == Approx(0.0));
    }

    vec3 self_field = pot_reci_inf.self_field({zeros, dipoles.at(1)});
    CHECK(self_field[0] == Approx(0.0));
    CHECK(self_field[1] == Approx(0.0));
    CHECK(self_field[2] == Approx(0.0));

    vec3 surface_field = pot_reci_inf.surface_field(positions, charges, dipoles, volume);
    CHECK(surface_field[0] == Approx(0.0));
    CHECK(surface_field[1] == Approx(0.0));
    CHECK(surface_field[2] == Approx(0.0));

    CHECK(real_field[0]+reciprocal_field[0]+self_field[0]+surface_field[0] == Approx(0.9956865));

    // ---- unit diel. coeff. of surrounding
    eps_sur = 1.0;
    Ewald pot_reci_unit(cutoff, alpha, eps_sur);

    surface_energy = pot_reci_unit.surface_energy(positions, charges, dipoles, volume);
    CHECK(surface_energy == Approx(0.0020943951));
    CHECK(real_energy+self_energy+surface_energy+reciprocal_energy == Approx(-1.0000311));

    surface_field = pot_reci_unit.surface_field(positions, charges, dipoles, volume);
    CHECK(surface_field[0] == Approx(0.0041887902));
    CHECK(surface_field[1] == Approx(0.0));
    CHECK(surface_field[2] == Approx(0.0));

    CHECK(real_field[0]+reciprocal_field[0]+self_field[0]+surface_field[0] == Approx(0.9998753));

    // -- two dipoles
    qA = 0.0;
    qB = 0.0;
    muA = {1.0, 0.0, 0.0};
    muB = {1.0, 0.0, 0.0};
    charges.at(0) = qA;
    charges.at(1) = qB;
    dipoles.at(0) = muA;
    dipoles.at(1) = muB;

    // ---- infinite diel. coeff. of surrounding

    real_energy = pot_reci_inf.dipole_dipole_energy(muA, muB, rAB);
    self_energy = pot_reci_inf.self_energy({qA*qA + qB*qB, muA.dot(muA) + muB.dot(muB)});
    surface_energy = pot_reci_inf.surface_energy(positions, charges, dipoles, volume);
    reciprocal_energy = pot_reci_inf.reciprocal_energy(positions,charges,dipoles,L,nmax);
    CHECK(real_energy == Approx(-2.0770407737));
    CHECK(self_energy == Approx(-0.3851534224));
    CHECK(surface_energy == Approx(0.0));
    CHECK(reciprocal_energy == Approx(0.4534417045));
    CHECK(real_energy+self_energy+surface_energy+reciprocal_energy == Approx(-2.0087525));

    real_field = pot_reci_inf.dipole_field(muB, rAB);
    CHECK(real_field[0] == Approx(2.0770407737 ));
    CHECK(real_field[1] == Approx(0.0));
    CHECK(real_field[2] == Approx(0.0));

    reciprocal_field = pot_reci_inf.reciprocal_field(positions, charges, dipoles, L, nmax, rA);
    CHECK(reciprocal_field[0] == Approx(-0.4534417045));
    CHECK(reciprocal_field[1] == Approx(0.0));
    CHECK(reciprocal_field[2] == Approx(0.0));

    self_field = pot_reci_inf.self_field({zeros, muB});
    CHECK(self_field[0] == Approx(0.3851534224));
    CHECK(self_field[1] == Approx(0.0));
    CHECK(self_field[2] == Approx(0.0));

    surface_field = pot_reci_inf.surface_field(positions, charges, dipoles, volume);
    CHECK(surface_field[0] == Approx(0.0));
    CHECK(surface_field[1] == Approx(0.0));
    CHECK(surface_field[2] == Approx(0.0));

    CHECK(real_field[0]+reciprocal_field[0]+self_field[0]+surface_field[0] == Approx(2.0087525));

    // ---- unit diel. coeff. of surrounding

    surface_energy = pot_reci_unit.surface_energy(positions, charges, dipoles, volume);
    CHECK(surface_energy == Approx(0.0083775804));
    CHECK(real_energy+self_energy+surface_energy+reciprocal_energy == Approx(-2.0003749));

    surface_field = pot_reci_unit.surface_field(positions, charges, dipoles, volume);
    CHECK(surface_field[0] == Approx(-0.0083775804));
    CHECK(surface_field[1] == Approx(0.0));
    CHECK(surface_field[2] == Approx(0.0));

    CHECK(real_field[0]+reciprocal_field[0]+self_field[0]+surface_field[0] == Approx(2.0003749));
}

TEST_CASE("[CoulombGalore] Ewald (truncated Gaussian) real-space") {
    using doctest::Approx;
    double cutoff = 29.0; // cutoff distance
    double alpha = 0.1;   // damping-parameter
    double eps_sur = infinity;
    EwaldT pot(cutoff, alpha, eps_sur);

    CHECK(pot.self_energy({4.0, 0.0}) == Approx(-0.2257993685));
    CHECK(pot.self_energy({0.0, 2.0}) == Approx(-0.0007528321650));

    // Test short-ranged function
    CHECK(pot.short_range_function(0.5) == Approx(0.03993035150));
    CHECK(pot.short_range_function_derivative(0.5) == Approx(-0.3992923727));
    CHECK(pot.short_range_function_second_derivative(0.5) == Approx(3.364180355));
    CHECK(pot.short_range_function_third_derivative(0.5) == Approx(-21.56439607));

    std::vector<double> charges = {1.0, -1.0, 2.0, 3.0};
    double volume = 2.0;
    CHECK(pot.neutralization_energy(charges, volume) == Approx(-1955.4692924725));

    testDerivatives(pot, 0.5); // Compare differentiation with numerical diff.
}

TEST_CASE("[CoulombGalore] Poisson") {
    using doctest::Approx;
    signed C = 3;         // number of cancelled derivatives at origin -2 (starting from second derivative)
    signed D = 3;         // number of cancelled derivatives at the cut-off (starting from zeroth derivative)
    double cutoff = 29.0; // cutoff distance
    Poisson pot33(cutoff, C, D);

    CHECK(pot33.self_energy({4.0, 0.0}) == Approx(-0.137931034482759));
    CHECK(pot33.self_energy({0.0, 2.0}) == Approx(0.0));

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

    testDerivatives(pot33, 0.5); // Compare differentiation with numerical diff.

    C = 4;                  // number of cancelled derivatives at origin -2 (starting from second derivative)
    D = 3;                  // number of cancelled derivatives at the cut-off (starting from zeroth derivative)
    double zA = 2.0;        // charge
    double zB = 3.0;        // charge
    vec3 muA = {19, 7, 11}; // dipole moment
    vec3 muB = {13, 17, 5}; // dipole moment
    mat33 quadA, quadB;     // quadrupole moment
    quadA << 3, 7, 8, 5, 9, 6, 2, 1, 4;
    quadB << 0, 0, 0, 0, 0, 0, 0, 0, 0;
    vec3 r = {23, 0, 0};    // distance vector
    vec3 rq = {5.75*std::sqrt(6.0), 5.75*std::sqrt(2.0), 11.5*std::sqrt(2.0)};    // distance vector for quadrupole check
    vec3 rh = {1, 0, 0};    // normalized distance vector
    Poisson pot43(cutoff, C, D);

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
    CHECK(pot43.quadrupole_potential(quadA, rq) == Approx(0.000899228165));
    CHECK(pot43.quadrupole_potential(quadA, rq/23.0*29.0) == Approx(0.0)); // at the cutoff

    // Test fields
    CHECK(pot43.ion_field(zA, cutoff * rh).norm() == Approx(0.0));
    vec3 E_ion = pot43.ion_field(zA, r);
    CHECK(E_ion[0] == Approx(0.0006052849004));
    CHECK(E_ion.norm() == Approx(0.0006052849004));
    CHECK(pot43.dipole_field(muA, cutoff * rh).norm() == Approx(0.0));
    vec3 E_dipole = pot43.dipole_field(muA, r);
    CHECK(E_dipole[0] == Approx(0.002702513754));
    CHECK(E_dipole[1] == Approx(-0.00009210857180));
    CHECK(E_dipole[2] == Approx(-0.0001447420414));
    vec3 E_quadrupole = pot43.quadrupole_field(quadA, r);
    CHECK(E_quadrupole[0] == Approx(0.00001919309993));
    CHECK(E_quadrupole[1] == Approx(-0.00004053806958));
    CHECK(E_quadrupole[2] == Approx(-0.00003378172465));
    vec3 E_multipole = pot43.multipole_field(zA,muA,quadA, r);
    CHECK(E_multipole[0] == Approx(0.003326991754330));
    CHECK(E_multipole[1] == Approx(-0.00013264664138));
    CHECK(E_multipole[2] == Approx(-0.00017852376605));

    // Test energies
    CHECK(pot43.ion_ion_energy(zA, zB, cutoff) == Approx(0.0));
    CHECK(pot43.ion_ion_energy(zA, zB, r.norm()) == Approx(0.002829195636));
    CHECK(pot43.ion_dipole_energy(zA, muB, cutoff * rh) == Approx(0.0));
    CHECK(pot43.ion_dipole_energy(zA, muB, r) == Approx(-0.007868703705));
    CHECK(pot43.ion_dipole_energy(zB, muA, -r) == Approx(0.01725061966));
    CHECK(pot43.dipole_dipole_energy(muA, muB, cutoff * rh) == Approx(0.0));
    CHECK(pot43.dipole_dipole_energy(muA, muB, r) == Approx(-0.03284312288));
    CHECK(pot43.ion_quadrupole_energy(zB, quadA, rq/23.0*29.0) == Approx(0.0));
    CHECK(pot43.ion_quadrupole_energy(zB, quadA, rq) == Approx(0.002697684495));
    CHECK(pot43.ion_quadrupole_energy(zA, quadB, -rq/23.0*29.0) == Approx(0.0));
    CHECK(pot43.ion_quadrupole_energy(zA, quadB, -rq) == Approx(0.0));
    CHECK(pot43.multipole_multipole_energy(zA, zB, muA, muB, quadA, quadB, r) == Approx(-0.020248530406300));

    // Test forces
    CHECK(pot43.ion_ion_force(zA, zB, cutoff * rh).norm() == Approx(0.0));
    vec3 F_ionion = pot43.ion_ion_force(zA, zB, r);
    CHECK(F_ionion[0] == Approx(0.001815854701));
    CHECK(F_ionion.norm() == Approx(0.001815854701));
    CHECK(pot43.ion_dipole_force(zB, muA, cutoff * rh).norm() == Approx(0.0));
    vec3 F_iondipoleBA = pot43.ion_dipole_force(zB, muA, r);
    CHECK(F_iondipoleBA[0] == Approx(0.008107541263));
    CHECK(F_iondipoleBA[1] == Approx(-0.0002763257154));
    CHECK(F_iondipoleBA[2] == Approx(-0.0004342261242));

    vec3 F_iondipoleAB = pot43.ion_dipole_force(zA, muB, -r);
    CHECK(F_iondipoleAB[0] == Approx(0.003698176716));
    CHECK(F_iondipoleAB[1] == Approx(-0.0004473844916));
    CHECK(F_iondipoleAB[2] == Approx(-0.0001315836740));

    CHECK(pot43.dipole_dipole_force(muA, muB, cutoff * rh).norm() == Approx(0.0));
    vec3 F_dipoledipole = pot43.dipole_dipole_force(muA, muB, r);
    CHECK(F_dipoledipole[0] == Approx(0.009216400961));
    CHECK(F_dipoledipole[1] == Approx(-0.002797126801));
    CHECK(F_dipoledipole[2] == Approx(-0.001608010094));

    vec3 F_multipolemultipole = pot43.multipole_multipole_force(zA, zB, muA, muB, quadA, quadB, r);
    CHECK(F_multipolemultipole[0] == Approx(0.022895552940790));
    CHECK(F_multipolemultipole[1] == Approx(-0.00364245121674));
    CHECK(F_multipolemultipole[2] == Approx(-0.00227516506615));

    // Test Yukawa-interactions
    C = 3;         // number of cancelled derivatives at origin -2 (starting from second derivative)
    D = 3;         // number of cancelled derivatives at the cut-off (starting from zeroth derivative)
    cutoff = 29.0; // cutoff distance
    double debye_length = 23.0;
    Poisson potY(cutoff, C, D, debye_length);

    CHECK(potY.self_energy({4.0, 0.0}) == Approx(-0.03037721287));
    CHECK(potY.self_energy({0.0, 2.0}) == Approx(0.0));

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
    CHECK(potY.quadrupole_potential(quadA, rq) == Approx(0.0016294707475));
    CHECK(potY.quadrupole_potential(quadA, rq/23.0*29.0) == Approx(0.0)); // at the cutoff

    // Test fields
    CHECK(potY.ion_field(zA, cutoff * rh).norm() == Approx(0.0));
    vec3 E_ion_Y = potY.ion_field(zA, r);
    CHECK(E_ion_Y[0] == Approx(0.001699041230));
    CHECK(E_ion_Y.norm() == Approx(0.001699041230));
    CHECK(potY.dipole_field(muA, cutoff * rh).norm() == Approx(0.0));
    vec3 E_dipole_Y = potY.dipole_field(muA, r);
    CHECK(E_dipole_Y[0] == Approx(0.004956265485));
    CHECK(E_dipole_Y[1] == Approx(-0.0002585497523));
    CHECK(E_dipole_Y[2] == Approx(-0.0004062924688));
    vec3 E_quadrupole_Y = potY.quadrupole_field(quadA, r);
    CHECK(E_quadrupole_Y[0] == Approx(-0.00005233355205));
    CHECK(E_quadrupole_Y[1] == Approx(-0.00007768480608));
    CHECK(E_quadrupole_Y[2] == Approx(-0.00006473733856));
    vec3 E_multipole_Y = potY.multipole_field(zA,muA,quadA, r);
    CHECK(E_multipole_Y[0] == Approx(0.006602973162950));
    CHECK(E_multipole_Y[1] == Approx(-0.00033623455838));
    CHECK(E_multipole_Y[2] == Approx(-0.00047102980736));

    // Test energies
    CHECK(potY.ion_ion_energy(zA, zB, cutoff) == Approx(0.0));
    CHECK(potY.ion_ion_energy(zA, zB, r.norm()) == Approx(0.01003265793));
    CHECK(potY.ion_dipole_energy(zA, muB, cutoff * rh) == Approx(0.0));
    CHECK(potY.ion_dipole_energy(zA, muB, r) == Approx(-0.02208753604));
    CHECK(potY.ion_dipole_energy(zB, muA, -r) == Approx(0.04842267505));
    CHECK(potY.dipole_dipole_energy(muA, muB, cutoff * rh) == Approx(0.0));
    CHECK(potY.dipole_dipole_energy(muA, muB, r) == Approx(-0.05800464321));
    CHECK(potY.ion_quadrupole_energy(zB, quadA, rq/23.0*29.0) == Approx(0.0));
    CHECK(potY.ion_quadrupole_energy(zB, quadA, rq) == Approx(0.004888412229));
    CHECK(potY.multipole_multipole_energy(zA, zB, muA, muB, quadA, quadB, r) == Approx(-0.0211832396518));

    // Test forces
    CHECK(potY.ion_ion_force(zA, zB, cutoff * rh).norm() == Approx(0.0));
    vec3 F_ionion_Y = potY.ion_ion_force(zA, zB, r);
    CHECK(F_ionion_Y[0] == Approx(0.005097123689));
    CHECK(F_ionion_Y.norm() == Approx(0.005097123689));
    CHECK(potY.ion_dipole_force(zB, muA, cutoff * rh).norm() == Approx(0.0));
    vec3 F_iondipoleBA_Y = potY.ion_dipole_force(zB, muA, r);
    CHECK(F_iondipoleBA_Y[0] == Approx(0.01486879646));
    CHECK(F_iondipoleBA_Y[1] == Approx(-0.0007756492577));
    CHECK(F_iondipoleBA_Y[2] == Approx(-0.001218877402));

    vec3 F_iondipoleAB_Y = potY.ion_dipole_force(zA, muB, -r);
    CHECK(F_iondipoleAB_Y[0] == Approx(0.006782258035));
    CHECK(F_iondipoleAB_Y[1] == Approx(-0.001255813082));
    CHECK(F_iondipoleAB_Y[2] == Approx(-0.0003693567885));

    CHECK(potY.dipole_dipole_force(muA, muB, cutoff * rh).norm() == Approx(0.0));
    vec3 F_dipoledipole_Y = potY.dipole_dipole_force(muA, muB, r);
    CHECK(F_dipoledipole_Y[0] == Approx(0.002987655323));
    CHECK(F_dipoledipole_Y[1] == Approx(-0.005360251624));
    CHECK(F_dipoledipole_Y[2] == Approx(-0.003081497314));

    vec3 F_multipolemultipole_Y = potY.multipole_multipole_force(zA, zB, muA, muB, quadA, quadB, r);
    CHECK(F_multipolemultipole_Y[0] == Approx(0.02957883285085));
    CHECK(F_multipolemultipole_Y[1] == Approx(-0.00762476838194));
    CHECK(F_multipolemultipole_Y[2] == Approx(-0.00486394352018));

    CHECK(Poisson(cutoff, 1, -1).short_range_function(0.5) == Approx(Plain().short_range_function(0.5) ));
    CHECK(Poisson(cutoff, 1, -1, debye_length).short_range_function(0.5) == Approx(Plain(debye_length).short_range_function(0.5) ));
    CHECK(Poisson(cutoff, 1, 0).short_range_function(0.5) == Approx(Wolf(cutoff,0.0).short_range_function(0.5) ));
}

TEST_CASE("[CoulombGalore] createScheme") {
    using doctest::Approx;
    double cutoff = 29.0;   // cutoff distance
    double zA = 2.0;        // charge
    vec3 muA = {19, 7, 11}; // dipole moment
    // vec3 muB = {13, 17, 5}; // dipole moment
    vec3 r = {23, 0, 0}; // distance vector
    vec3 rh = {1, 0, 0}; // normalized distance vector

#ifdef NLOHMANN_JSON_HPP
    // create scheme dynamically through a json object
    auto pot = createScheme({{"type", "plain"}});

    // Test potentials
    CHECK(pot->ion_potential(zA, cutoff + 1.0) == Approx(0.06666666667));
    CHECK(pot->ion_potential(zA, r.norm()) == Approx(0.08695652174));
    CHECK(pot->dipole_potential(muA, (cutoff + 1.0) * rh) == Approx(0.02111111111));
    CHECK(pot->dipole_potential(muA, r) == Approx(0.03591682420));

    // Test Yukawa-interaction and note that we merely re-assign `pot`
    pot = createScheme({{"type", "poisson"}, {"cutoff", cutoff}, {"C", 3}, {"D", 3}, {"debyelength", 23}});
    CHECK(pot->ion_potential(zA, cutoff) == Approx(0.0));
    CHECK(pot->ion_potential(zA, r.norm()) == Approx(0.003344219306));

    pot = createScheme(
        nlohmann::json({{"type", "poisson"}, {"cutoff", cutoff}, {"C", 3}, {"D", 3}, {"debyelength", 23}}));
    CHECK(pot->ion_potential(zA, cutoff) == Approx(0.0));
    CHECK(pot->ion_potential(zA, r.norm()) == Approx(0.003344219306));
#endif
}

TEST_CASE("[CoulombGalore] Splined") {
    using doctest::Approx;
    double tol = 0.001;   // tolerance
    double cutoff = 29.0; // cutoff distance
    Splined pot;

    SUBCASE("Plain") {
        double zA = 2.0;        // charge
        double zB = 3.0;        // charge
        vec3 muA = {19, 7, 11}; // dipole moment
        vec3 muB = {13, 17, 5}; // dipole moment
        vec3 r = {23, 0, 0};    // distance vector
        vec3 rh = {1, 0, 0};    // normalized distance vector
        pot.spline<Plain>();

        // Test self energy
        CHECK(pot.self_energy({zA * zB, muA.norm() * muB.norm()}) == Approx(0.0));

        // Test short-ranged function
        CHECK(pot.short_range_function(0.5) == Approx(1.0));
        CHECK(pot.short_range_function_derivative(0.5) == Approx(0.0));
        CHECK(pot.short_range_function_second_derivative(0.5) == Approx(0.0));
        CHECK(pot.short_range_function_third_derivative(0.5) == Approx(0.0));

        testDerivatives(pot, 0.5); // Compare differentiation with numerical diff.

        // Test potentials
        CHECK(pot.ion_potential(zA, cutoff + 1.0) == Approx(0.06666666667));
        CHECK(pot.ion_potential(zA, r.norm()) == Approx(0.08695652174));
        CHECK(pot.dipole_potential(muA, (cutoff + 1.0) * rh) == Approx(0.02111111111));
        CHECK(pot.dipole_potential(muA, r) == Approx(0.03591682420));

        // Test energies
        CHECK(pot.ion_ion_energy(zA, zB, (cutoff + 1.0)) == Approx(0.2));
        CHECK(pot.ion_ion_energy(zA, zB, r.norm()) == Approx(0.2608695652));
        CHECK(pot.ion_dipole_energy(zA, muB, (cutoff + 1.0) * rh) == Approx(-0.02888888889));
        CHECK(pot.ion_dipole_energy(zA, muB, r) == Approx(-0.04914933837));
        CHECK(pot.dipole_dipole_energy(muA, muB, (cutoff + 1.0) * rh) == Approx(-0.01185185185));
        CHECK(pot.dipole_dipole_energy(muA, muB, r) == Approx(-0.02630064930));

        // Test forces
        CHECK(pot.ion_ion_force(zA, zB, (cutoff + 1.0) * rh).norm() == Approx(0.006666666667));
        vec3 F_ionion = pot.ion_ion_force(zA, zB, r);
        CHECK(F_ionion[0] == Approx(0.01134215501));
        CHECK(F_ionion.norm() == Approx(0.01134215501));
        CHECK(pot.ion_dipole_force(zB, muA, (cutoff + 1.0) * rh).norm() == Approx(0.004463846540));
        vec3 F_iondipole = pot.ion_dipole_force(zB, muA, r);
        CHECK(F_iondipole[0] == Approx(0.009369606312));
        CHECK(F_iondipole[1] == Approx(-0.001725980110));
        CHECK(F_iondipole[2] == Approx(-0.002712254459));
        CHECK(pot.dipole_dipole_force(muA, muB, (cutoff + 1.0) * rh).norm() == Approx(0.002129033733));
        vec3 F_dipoledipole = pot.dipole_dipole_force(muA, muB, r);
        CHECK(F_dipoledipole[0] == Approx(0.003430519474));
        CHECK(F_dipoledipole[1] == Approx(-0.004438234569));
        CHECK(F_dipoledipole[2] == Approx(-0.002551448858));
    }

    SUBCASE("Wolf") {
        double alpha = 0.1; // damping-parameter
        pot.spline<Wolf>(alpha, cutoff);
        CHECK(pot.short_range_function(0.5) == Approx(0.04028442542).epsilon(tol));
        CHECK(pot.short_range_function_derivative(0.5) == Approx(-0.3997546829).epsilon(tol));
        CHECK(pot.short_range_function_second_derivative(0.5) == Approx(3.36159125).epsilon(tol));
        CHECK(pot.short_range_function_third_derivative(0.5) == Approx(-21.54779991).epsilon(tol));
        CHECK(pot.short_range_function(1.0) == Approx(0.0).epsilon(tol));
        CHECK(pot.self_energy({4.0, 0.0}) == Approx(Wolf(alpha, cutoff).self_energy({4.0, 0.0})));
    }

    SUBCASE("qPotential") {
        int order = 3;
        pot.spline<qPotential>(cutoff, order);
        // Test self energy
        CHECK(pot.self_energy({4.0, 0.0}) == Approx(-0.06896551724));
        CHECK(pot.self_energy({0.0, 2.0}) == Approx(-0.000041002091105));
    }

    SUBCASE("Poisson") {
        int C = 4;              // number of cancelled derivatives at origin -2 (starting from second derivative)
        int D = 3;              // number of cancelled derivatives at the cut-off (starting from zeroth derivative)
        double zA = 2.0;        // charge
        vec3 muA = {19, 7, 11}; // dipole moment
        vec3 r = {0.5 * cutoff, 0, 0}; // distance vector
        vec3 rh = {1, 0, 0};           // normalized distance vector

        pot.spline<Poisson>(cutoff, C, D);
        CHECK(pot.name == "poisson");
        CHECK(pot.scheme == Scheme::poisson);

        // Test short-ranged function
        CHECK(pot.short_range_function(0.5) == Approx(0.19921875).epsilon(tol));
        CHECK(pot.short_range_function_derivative(0.5) == Approx(-1.1484375).epsilon(tol));
        CHECK(pot.short_range_function_second_derivative(0.5) == Approx(3.28125).epsilon(tol));
        CHECK(pot.short_range_function_third_derivative(0.5) == Approx(6.5625).epsilon(tol));

        // Test potentials
        CHECK(pot.ion_potential(zA, cutoff) == Approx(0.0).epsilon(tol));
        CHECK(pot.ion_potential(zA, r.norm()) == Approx(0.02747844828).epsilon(tol));

        CHECK((cutoff * rh).norm() == Approx(r.norm() * 2.0).epsilon(tol));
        CHECK(pot.dipole_potential(muA, cutoff * rh) == Approx(0.0).epsilon(tol));
        CHECK(pot.dipole_potential(muA, r) == Approx(0.06989447087).epsilon(tol));
    }

    SUBCASE("Ewald") {
        double cutoff = 29.0;        // cutoff distance
        double alpha = 0.1;          // damping-parameter
        double epsr_surf = infinity; // surface dielectric

        pot.spline<Ewald>(cutoff, alpha, epsr_surf);
        CHECK(pot.name == "Ewald real-space");
        CHECK(pot.scheme == Scheme::ewald);

        // Test short-ranged function
        CHECK(pot.short_range_function(0.5) == Approx(0.04030497436).epsilon(tol));
        CHECK(pot.short_range_function_derivative(0.5) == Approx(-0.399713585).epsilon(tol));
        CHECK(pot.short_range_function_second_derivative(0.5) == Approx(3.36159125).epsilon(tol));
        CHECK(pot.short_range_function_third_derivative(0.5) == Approx(-21.54779991).epsilon(tol));

        double debyelength = 23.0;
        pot.spline<Ewald>(cutoff, alpha, epsr_surf, debyelength);

        // Test short-ranged function
        CHECK(pot.short_range_function(0.5) == Approx(0.07306333588).epsilon(tol));
        CHECK(pot.short_range_function_derivative(0.5) == Approx(-0.63444119).epsilon(tol));
        CHECK(pot.short_range_function_second_derivative(0.5) == Approx(4.423133599).epsilon(tol));
        CHECK(pot.short_range_function_third_derivative(0.5) == Approx(-19.85937171).epsilon(tol));
    }
}
