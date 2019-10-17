# Coulomb Galore

This is a C++ library for calculating the potential, field, forces, and interactions from and between electric multipoles. Focus is on approximate truncation schemes that offer fast alternatives to Ewald summation. All implemented methods are unit tested.

## Usage

### Requirements

- C++14 compiler
- The Eigen matrix library
- nlohmann::json (optional)
- doctest (optional)

### Building

The CMake build will automatically download Eigen, json, and doctest.

~~~ bash
cmake .
make
make test
~~~

### Use in your own code

Simply copy the `coulombgalore.h` file to your project. All functions and classes are encapsulated in the `CoulombGalore` namespace. Vectors are currently handled by the Eigen library, but it is straightforward to change to another library.

### Simple example

~~~ cpp
#include <iostream>
#include "coulombgalore.h"
using namespace 
int main() {
   Eigen::Vector3d R = {0,0,10};                      // distance vector
   CoulombGalore::Plain pot(14.0);                    // cutoff distance as constructor argument
   double u = pot.ion_ion_energy(1.0, 1.0, R.norm()); // potential energy = 1.0*1.0/10

   Eigen::Vector3d mu = {2,5,2};                      // dipole moment
   Eigen::Vector3d E = pot.dipole_field(mu, R);       // field from dipole at 𝐑
}
~~~
