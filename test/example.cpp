#include <iostream>
#include <nlohmann/json.hpp>
#include "coulombgalore.h"

using namespace CoulombGalore;

typedef Eigen::Vector3d Point; //!< typedef for 3d vector

int main() {
    double pi = 3.141592653589793, // Pi
        e0 = 8.85419e-12,          // Permittivity of vacuum [C^2/(J*m)]
        e = 1.602177e-19,          // Absolute electronic unit charge [C]
        T = 298.15,                // Temperature [K]
        kB = 1.380658e-23;         // Boltzmann's constant [J/K]

    double z1 = 1, z2 = 2;    // two monopoles
    double cutoff = 18e-10;   // cutoff distance, here in meters [m]
    vec3 r = {7.0e-10, 0, 0}; // a distance vector, use same using as cutoff, i.e. [m]

    // energies are returned in electrostatic units and we must multiply
    // with the Coulombic constant to get more familiar units:
    double bjerrum_length = e * e / (4 * pi * e0 * kB * T); // [m]

    // this is just the plain old Coulomb potential
    Plain pot_plain;
    double u12 = pot_plain.ion_ion_energy(z1, z2, r.norm());
    std::cout << "plain ion-ion energy:      " << bjerrum_length * u12 << " kT" << std::endl;

    // this is just the plain old Coulomb potential
    double debye_length = 23.01e-10;
    Plain pot_plainY(debye_length);
    u12 = pot_plainY.ion_ion_energy(z1, z2, r.norm());
    std::cout << "plain ion-ion energy:      " << bjerrum_length * u12 << " kT" << std::endl;

    // this is a truncated potential
    qPotential pot_qpot(cutoff, 3);
    u12 = pot_qpot.ion_ion_energy(z1, z2, r.norm());
    std::cout << "qPotential ion-ion energy: " << bjerrum_length * u12 << " kT" << std::endl;

    qPotential pot_qpot3(cutoff, 3);
    qPotential pot_qpot4(cutoff, 4);

    Fanourgakis pot_kis(cutoff);
    Ewald pot_ewald(cutoff, 0.1e10, infinity);

    for (double q = 0; q <= 1; q += 0.01)
        std::cout << q << " " << pot_qpot3.short_range_function(q) << " " << pot_qpot4.short_range_function(q) << " "
                  << " " << pot_kis.short_range_function(q) << " "
                  << " " << pot_ewald.short_range_function(q) << "\n";

#ifdef NLOHMANN_JSON_HPP
    // this is a truncated potential initiated using JSON
    Wolf pot_wolf(nlohmann::json({{"cutoff", cutoff}, {"alpha", 0.5}}));

    // if available, json can be used to (de)serialize
    nlohmann::json j;
    pot_qpot.to_json(j);
    std::cout << j << std::endl;
#endif
}
