#include <nlohmann/json.hpp>
#include "coulombgalore.h"
#include <iostream>

using namespace CoulombGalore;

int main() {
    double pi = 3.141592653589793, // Pi
        e0 = 8.85419e-12,          // Permittivity of vacuum [C^2/(J*m)]
        e = 1.602177e-19,          // Absolute electronic unit charge [C]
        T = 298.15,                // Temperature [K]
        kB = 1.380658e-23;         // Boltzmann's constant [J/K]

    double z1 = 1, z2 = 2;     // two monopoles
    double cutoff = 18e-10;    // cutoff distance, here in meters [m]
    Point r = {7.0e-10, 0, 0}; // a distance vector, use same using as cutoff, i.e. [m]

    // energies are returned in electrostatic units and we must multiply
    // with the Coulombic constant to get more familiar units:
    double bjerrum_length = e * e / (4 * pi * e0 * kB * T); // [m]

    // this is just the plain old Coulomb potential
    PairPotential<Plain> pot_plain;
    double u12 = pot_plain.ion_ion_energy(z1, z2, r.norm());
    std::cout << "plain ion-ion energy:      " << bjerrum_length * u12 << " kT" << std::endl;

    // this is a truncated potential
    PairPotential<qPotential> pot_qpot(cutoff, 3);
    u12 = pot_qpot.ion_ion_energy(z1, z2, r.norm());
    std::cout << "qPotential ion-ion energy: " << bjerrum_length * u12 << " kT" << std::endl;

    // this is a truncated potential initiated using JSON
    PairPotential<Wolf> pot_wolf( nlohmann::json({{"cutoff",cutoff}, {"alpha",0.5}}) );

#ifdef NLOHMANN_JSON_HPP
    // if available, json can be used to (de)serialize
    nlohmann::json j;
    pot_qpot.to_json(j);
    std::cout << j << std::endl;
#endif
}
