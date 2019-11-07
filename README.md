[![Documentation](https://codedocs.xyz/mlund/coulombgalore.svg)](https://codedocs.xyz/mlund/coulombgalore/)

# Coulomb Galore

This is a C++ library for calculating the potential, field, forces, and interactions from and between electric multipoles.
Focus is on approximate truncation schemes that offer fast alternatives to Ewald summation. All implemented methods are unit tested.

## Usage

### Requirements

- C++14 compiler
- The Eigen matrix library
- nlohmann::json (optional)
- doctest (optional)
- doxygen (optional, for building API manual)

### Building

The CMake build will automatically download Eigen, json, and doctest.

~~~ bash
cmake .
make
make test (optional)
doxygen (optional)
~~~

### Use in your own code

Simply copy the `coulombgalore.h` file to your project. All functions and classes are encapsulated in the `CoulombGalore` namespace. Vectors are currently handled by the Eigen library, but it is straightforward to change to another library.

### Example

~~~{.cpp}
#include "coulombgalore.h"
int main() {
   Eigen::Vector3d R = {0,0,10};                      // distance vector
   CoulombGalore::Plain pot(14.0);                    // cutoff distance as constructor argument
   double u = pot.ion_ion_energy(1.0, 1.0, R.norm()); // potential energy = 1.0*1.0/10

   Eigen::Vector3d mu = {2,5,2};                      // dipole moment
   Eigen::Vector3d E = pot.dipole_field(mu, R);       // field from dipole at 𝐑
}
~~~

### Available Truncation Schemes

Class name                                      | _S(q)_
----------------------------------------------- | ------------------------
[`Plain`](http://doi.org/ctnnsj)                | 1
[`ReactionField`](http://doi.org/10/dscmwg)     | ![equation](https://latex.codecogs.com/svg.latex?1&plus;%5Cfrac%7B%5Cepsilon_%7BRF%7D-%5Cepsilon_r%7D%7B2%5Cepsilon_%7BRF%7D&plus;%5Cepsilon_r%7Dq%5E3-3%5Cfrac%7B%5Cepsilon_%7BRF%7D%7D%7B2%5Cepsilon_%7BRF%7D&plus;%5Cepsilon_r%7Dq)
[`Poisson`](http://doi.org/10/c5fr)             | ![equation](https://latex.codecogs.com/svg.latex?%281-%5Ctilde%7Bq%7D%29%5E%7BD&plus;1%7D%5Csum_%7Bc%3D0%7D%5E%7BC-1%7D%5Cfrac%7BC-c%7D%7BC%7D%7BD-1&plus;c%5Cchoose%20c%7D%5Ctilde%7Bq%7D%5Ec)
[`qPotential`](https://arxiv.org/abs/1904.10335)| ![equation](https://latex.codecogs.com/svg.latex?%5Cprod_%7Bn%3D1%7D%5E%7B%5Ctext%7Border%7D%7D%281-q%5En%29)
[`Fanourgakis`](http://doi.org/f639q5)          | ![equation](https://latex.codecogs.com/svg.latex?1-%5Cfrac%7B7%7D%7B4%7Dq&plus;%5Cfrac%7B21%7D%7B4%7Dq%5E5-7q%5E6&plus;%5Cfrac%7B5%7D%7B2%7Dq%5E7)
[`Fennell`](http://doi.org/10.1063/1.2206581)   | ![equation](https://latex.codecogs.com/svg.latex?%5Ctext%7Berfc%7D%28%5Ceta%20q%29-q%5Ctext%7Berfc%7D%28%5Ceta%29&plus;%28q-1%29q%5Cleft%28%5Ctext%7Berfc%7D%28%5Ceta%29&plus;%5Cfrac%7B2%5Ceta%7D%7B%5Csqrt%7B%5Cpi%7D%7D%5Ctext%7Bexp%7D%28-%5Ceta%5E2%29%5Cright%29)
[`ZeroDipole`](http://doi.org/10.1063/1.3582791)| ![equation](https://latex.codecogs.com/svg.latex?%5Ctext%7Berfc%7D%28%5Ceta%20q%29-q%5Ctext%7Berfc%7D%28%5Ceta%29&plus;%5Cfrac%7B%28q%5E2-1%29%7D%7B2%7Dq%5Cleft%28%5Ctext%7Berfc%7D%28%5Ceta%29&plus;%5Cfrac%7B2%5Ceta%7D%7B%5Csqrt%7B%5Cpi%7D%7D%5Ctext%7Bexp%7D%28-%5Ceta%5E2%29%5Cright%29)
[`Zahn`](http://doi.org/10.1021/jp025949h)      | ![equation](https://latex.codecogs.com/svg.latex?%5Ctext%7Berfc%7D%28%5Ceta%20q%29-%28q-1%29q%5Cleft%28%5Ctext%7Berfc%7D%28%5Ceta%29&plus;%5Cfrac%7B2%5Ceta%7D%7B%5Csqrt%7B%5Cpi%7D%7D%5Ctext%7Bexp%7D%28-%5Ceta%5E2%29%5Cright%29)
[`Wolf`](http://doi.org/cfcxdk)                 | ![equation](https://latex.codecogs.com/svg.latex?%5Ctext%7Berfc%7D%28%5Ceta%20q%29-%5Ctext%7Berfc%7D%28%5Ceta%29q)
[`Ewald`](http://doi.org/dgpdmc)                | ![equation](https://latex.codecogs.com/svg.latex?%5Cfrac%7B1%7D%7B2%7D%5Ctext%7Berfc%7D%5Cleft%28%5Ceta%20q%20&plus;%20%5Cfrac%7B%5Ckappa%5E*%7D%7B2%5Ceta%7D%5Cright%29%5Ctext%7Bexp%7D%5Cleft%282%5Ckappa%5E*%20q%5Cright%29%20&plus;%20%5Cfrac%7B1%7D%7B2%7D%5Ctext%7Berfc%7D%5Cleft%28%5Ceta%20q%20-%20%5Cfrac%7B%5Ckappa%5E*%7D%7B2%5Ceta%7D%5Cright%29)
`Splined`                                       | Splined version of any of the above

Here 

![equation](https://latex.codecogs.com/svg.latex?q%3D%5Cfrac%7Br%7D%7BR_c%7D%5Cquad%5Cquad%20%5Ctilde%7Bq%7D%3D%5Cfrac%7B1-%5Cexp%282%5Ckappa%5E*q%29%7D%7B1-%5Cexp%282%5Ckappa%5E*%29%7D%20%5Cquad%5Cquad%20%5Ceta%20%3D%20%5Calpha%20R_c%20%5Cquad%5Cquad%20%5Ckappa%5E*%3D%5Ckappa%20R_c.) 

### Units

It is vital that the units of the input parameters and function input values are consistent, such that correct output units are retrieved.
In terms of the charge unit `Z`, and length unit `L`, the input parameters and function outputs are listed in tables below.
All charges must have units `Z`, dipoles `Z*L`, distances `L`, volumes `L^3`, and fields `Z/L^2`.
Also note that the input `M2V` for function `calc_dielectric` has to be unitless.

Input parameter | Unit
--------------- | -------------------
`cutoff`        | `L`
`debye_length`  | `L`
`alpha`         | `L^-1`
`order`         | `positive integer`
`C`             | `positive integer`
`D`             | `integer`
`epss`          | `unitless`
`epsRF`         | `unitless`
`epsr`          | `unitless`
`shifted`       | `boolean`


Function                    | Output unit
--------------------------- | -------------
`ion_potential`             | `Z / L`
`dipole_potential`          | `Z / L`
`ion_field`                 | `Z / L^2`
`dipole_field`              | `Z / L^2`
`multipole_field`           | `Z / L^2`
`ion_ion_energy`            | `Z^2 / L`
`ion_dipole_energy`         | `Z^2 / L`
`dipole_dipole_energy`      | `Z^2 / L`
`multipole_multipole_energy`| `Z^2 / L`
`ion_ion_force`             | `Z^2 / L^2`
`ion_dipole_force`          | `Z^2 / L^2`
`dipole_dipole_force`       | `Z^2 / L^2`
`multipole_multipole_force` | `Z^2 / L^2`
`dipole_torque`             | `Z^2 / L`
`self_energy`               | `Z^2 / L`
`neutralization_energy`     | `Z^2 / L`
