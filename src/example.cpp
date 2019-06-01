#include "coulomb.h"
#include <iostream>

using namespace CoulombGalore;

int main() {
    double pi = 3.141592653589793, // Pi
        e0 = 8.85419e-12,          // Permittivity of vacuum [C^2/(J*m)]
        e = 1.602177e-19,          // Absolute electronic unit charge [C]
        T = 298.15,                // Temperature [K]
        kB = 1.380658e-23;         // Boltzmann's constant [J/K]

    double z1 = 1, z2 = 2;     // two monopoles
    Point r = {7.0e-10, 0, 0}; // a distance vector (meters)

    // energies are returned in electrostatic units and we must multiply
    // with the Coulombic to get more familiar units:
    double bjerrum_length = e * e / (4 * pi * e0 * kB * T); // meters

    PairPotential<Plain> pot;
    std::cout << "ion-ion energy: " << bjerrum_length * pot.ion_ion(z1 * z2, r.norm()) << " kT" << std::endl;
}
