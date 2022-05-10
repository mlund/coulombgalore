/*
* CoulombGalore - A Library for Electrostatic Interactions
*
* MIT License
* Copyright (c) 2019 Björn Stenqvist and Mikael Lund
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
#pragma once

#include "core.h"

namespace CoulombGalore {

#ifdef NLOHMANN_JSON_HPP
NLOHMANN_JSON_SERIALIZE_ENUM(ReciprocalEwaldState::Policies, {
                                                                 {ReciprocalEwaldState::INVALID, nullptr},
                                                                 {ReciprocalEwaldState::PBC, "PBC"},
                                                                 {ReciprocalEwaldState::PBCEigen, "PBCEigen"},
                                                                 {ReciprocalEwaldState::IPBC, "IPBC"},
                                                                 {ReciprocalEwaldState::IPBCEigen, "IPBCEigen"},
                                                             })
#endif

// -------------- Ewald real-space (using Gaussian) ---------------

/**
 * @brief Data class for Ewald k-space calculations
 *
 * This contains the _state_ of reciprocal Ewald algorithms.
 * Policies are currently not in use.
 *
 * Related reading:
 * - PBC Ewald (DOI:10.1063/1.481216)
 * - IPBC Ewald (DOI:10/css8)
 * - Update optimization (DOI:10.1063/1.481216, Eq. 24)
 */
class ReciprocalEwaldState {
  protected:
    const double pi = 4.0 * std::atan(1.0);
    Eigen::Matrix3Xd k_vectors; //!< k-vectors, 3xK
    Eigen::VectorXd Aks;        //!< 1xK for update optimization (see Eq.24, DOI:10.1063/1.481216)
    Eigen::VectorXcd Q_mp;      //!< Complex 1xK vectors

  public:
    int reciprocal_cutoff = 0.0;              //!< Inverse space cutoff
    double cutoff = 0.0;                      //!< Real-space cutoff
    double surface_dielectric_constant = 0.0; //!< Surface dielectric constant;
    double kappa = 0.0;                       //!< Inverse Debye screening length
    double alpha = 0.0;
    vec3 box_length = {0.0, 0.0, 0.0};                         //!< Box dimensions
    enum Policies { PBC, PBCEigen, IPBC, IPBCEigen, INVALID }; //!< Possible k-space updating schemes
    Policies policy = PBC;                                     //!< Policy for updating k-space

    inline int numKVectors() const { return k_vectors.cols(); }
    inline auto getVolume() const { return box_length.prod(); }
    inline void setZeroComplex() { Q_mp.setZero(); }

    /**
     * @brief Resize k-vectors, Q, and Aks. Present values are conserved.
     * @param number_of_k_vectors Number of k-vectors
     * @todo Is the special case if zero really needed?
     */
    void resize(int number_of_k_vectors) {
        if (number_of_k_vectors == 0) {
            resize(1);
            k_vectors.col(0) = vec3(1.0, 0.0, 0.0); // Just so it is not the zero-vector
            Aks.setZero();
        } else {
            k_vectors.conservativeResize(3, number_of_k_vectors);
            Q_mp.conservativeResize(number_of_k_vectors);
            Aks.conservativeResize(number_of_k_vectors);
        }
    }
    template <typename Positions, typename Charges, typename Dipoles>
    vec3 dipoleMoment(const Positions &positions, const Charges &charges, const Dipoles &dipoles) const {
        vec3 sum = {0.0, 0.0, 0.0};
        auto charge = charges.begin();
        for (const auto &position : positions) {
            sum += position * (*charge);
            charge++;
        }
        for (const auto &dipole : dipoles) {
            sum += dipole;
        }
        return sum;
    }
};

/**
 * @brief Ewald real-space scheme using a Gaussian screening-function.
 *
 * @note The implemented charge-compensation for Ewald differes from that of in DOI:10.1021/ct400626b where chi = -pi /
 * alpha2. This expression is only correct if integration is over all space, not just the cutoff region, cf. Eq. 14
 * in 10.1021/jp951011v. Thus the implemented expression is roughly -pi / alpha2 for alpha > ~2-3. User beware!
 * (also see DOI:10.1063/1.470721)
 */
class Ewald : public EnergyImplementation<Ewald> {
    double eta, eta2, eta3;             //!< Reduced damping-parameter, and squared, and cubed
    double zeta, zeta2, zeta3;          //!< Reduced inverse Debye-length, and squared, and cubed
    double surface_dielectric_constant; //!< Dielectric constant of the surrounding medium
    const double pi_sqrt = 2.0 * std::sqrt(std::atan(1.0));
    const double pi = 4.0 * std::atan(1.0);

  public:
    inline Ewald(const ReciprocalEwaldState &state)
        : Ewald(state.cutoff, state.alpha, state.surface_dielectric_constant, 1.0 / state.kappa) {}

    /**
     * @param cutoff distance cutoff
     * @param alpha damping-parameter
     */
    inline Ewald(double cutoff, double alpha, double surface_dielectric_constant = infinity,
                 double debye_length = infinity)
        : EnergyImplementation(Scheme::ewald, cutoff), surface_dielectric_constant(surface_dielectric_constant) {
        name = "Ewald real-space";
        has_dipolar_selfenergy = true;
        doi = "10.1002/andp.19213690304";
        eta = alpha * cutoff;
        eta2 = eta * eta;
        eta3 = eta2 * eta;
        if (surface_dielectric_constant < 1.0) {
            surface_dielectric_constant = infinity;
        }

        // Eq. 12 in DOI: 10.1016/0009-2614(83)80585-5 using 'K = cutoff region'
        const auto Q = 1.0 - std::erfc(eta) - 2.0 * eta / pi_sqrt * std::exp(-eta2);
        if (std::isinf(surface_dielectric_constant)) {
            T0 = Q;
        } else { // Eq. 17 in DOI: 10.1016/0009-2614(83)80585-5
            T0 = Q - 1.0 + 2.0 * (surface_dielectric_constant - 1.0) / (2.0 * surface_dielectric_constant + 1.0);
        }
        zeta = cutoff / debye_length;
        zeta2 = zeta * zeta;
        zeta3 = zeta2 * zeta;

        // if close to zero the general expression numerically diverges, and this expresion is used instead
        if (zeta < 1e-6) {
            chi = -pi * cutoff_squared *
                  (1.0 - std::erfc(eta) * (1.0 - 2.0 * eta2) - 2.0 * eta * std::exp(-eta2) / pi_sqrt) / eta2;
        } else {
            chi = 4.0 *
                  (0.5 * (1.0 - zeta) * std::erfc(eta + zeta / (2.0 * eta)) * std::exp(zeta) +
                   std::erf(eta) * std::exp(-zeta2 / (4.0 * eta2)) +
                   0.5 * (1.0 + zeta) * std::erfc(eta - zeta / (2.0 * eta)) * std::exp(-zeta) - 1.0) *
                  pi * cutoff_squared / zeta2;
        }
        // chi = -pi * cutoff_squared / eta2 according to DOI:10.1021/ct400626b, for uncscreened system

        setSelfEnergyPrefactor(
            {-eta / pi_sqrt *
                 (std::exp(-zeta2 / 4.0 / eta2) - pi_sqrt * zeta / (2.0 * eta) * std::erfc(zeta / (2.0 * eta))),
             -eta3 / pi_sqrt * 2.0 / 3.0 *
                 (pi_sqrt * zeta3 / 4.0 / eta3 * std::erfc(zeta / (2.0 * eta)) +
                  (1.0 - zeta2 / 2.0 / eta2) * std::exp(-zeta2 / 4.0 / eta2))}); // ion-quadrupole self-energy term: XYZ
        setSelfFieldPrefactor({0.0, 4.0 / 3.0 * eta3 / pi_sqrt});
    }

    inline double short_range_function(double q) const override {
        return 0.5 * (std::erfc(eta * q + zeta / (2.0 * eta)) * std::exp(2.0 * zeta * q) +
                      std::erfc(eta * q - zeta / (2.0 * eta)));
    }
    inline double short_range_function_derivative(double q) const override {
        const auto expC = std::exp(-powi(eta * q - zeta / (2.0 * eta), 2));
        const auto erfcC = std::erfc(eta * q + zeta / (2.0 * eta));
        return (-2.0 * eta / pi_sqrt * expC + zeta * erfcC * std::exp(2.0 * zeta * q));
    }
    inline double short_range_function_second_derivative(double q) const override {
        const auto expC = std::exp(-powi(eta * q - zeta / (2.0 * eta), 2));
        const auto erfcC = std::erfc(eta * q + zeta / (2.0 * eta));
        return (4.0 * eta2 / pi_sqrt * (eta * q - zeta / eta) * expC + 2.0 * zeta2 * erfcC * std::exp(2.0 * zeta * q));
    }
    inline double short_range_function_third_derivative(double q) const override {
        const auto expC = std::exp(-powi(eta * q - zeta / (2.0 * eta), 2));
        const auto erfcC = std::erfc(eta * q + zeta / (2.0 * eta));
        return (4.0 * eta3 / pi_sqrt *
                    (1.0 - 2.0 * (eta * q - zeta / eta) * (eta * q - zeta / (2.0 * eta)) - zeta2 / eta2) * expC +
                4.0 * zeta3 * erfcC * std::exp(2.0 * zeta * q));
    }

#ifdef NLOHMANN_JSON_HPP
    inline Ewald(const nlohmann::json &j)
        : Ewald(j.at("cutoff").get<double>(), j.at("alpha").get<double>(), j.value("epss", infinity),
                j.value("debyelength", infinity)) {}

  private:
    inline void _to_json(nlohmann::json &j) const override {
        j = {{"alpha", eta / cutoff}};
        if (std::isinf(surface_dielectric_constant)) {
            j["epss"] = "inf";
        } else {
            j["epss"] = surface_dielectric_constant;
        }
    }
#endif
};

class ReciprocalEwaldGaussian : public ReciprocalEwaldState {
  private:
    void setAks() {
        const auto cutoff_squared = cutoff * cutoff;
        const auto reduced_damping_squared = alpha * alpha * cutoff_squared;
        const auto reduced_kappa_squared = kappa * kappa * cutoff_squared;
        for (int i = 0; i < Aks.size(); i++) {
            const double k2 = k_vectors.col(i).squaredNorm() + kappa * kappa;
            Aks[i] = std::exp(-(k2 * cutoff_squared + reduced_kappa_squared) / (4.0 * reduced_damping_squared)) / k2;
        }
    }

    /**
     * @brief Calculates Q value
     * @param positions Positions of particles
     * @param charges Charges of particles (must have same length as positions)
     * @param dipoles Dipole moments of particles (must have same length as positions)
     * @param k Single k-vector
     * @todo May be optimized to splitting into several loops for cos/sin, enabling SIMD optimization
     */
    template <typename Positions, typename Charges, typename Dipoles>
    auto calcQ(const vec3 &k, const Positions &positions, const Charges &charges, const Dipoles &dipoles) const {
        Tcomplex Q(0.0, 0.0);
        auto charge = charges.begin();
        auto dipole = dipoles.begin();
        std::for_each(positions.begin(), positions.end(), [&](const auto &position) {
            const auto kr = k.dot(position); // 𝒌⋅𝒓
            const auto cos_kr = std::cos(kr);
            const auto sin_kr = std::sin(kr);
            Q += (*charge) * Tcomplex(cos_kr, sin_kr) + (*dipole).dot(k) * Tcomplex(-sin_kr, cos_kr);
            charge++;
            dipole++;
        });
        return Q;
    }

  public:
    const Ewald real_space; //!< Real-space energy functions

    explicit ReciprocalEwaldGaussian(const ReciprocalEwaldState &state)
        : ReciprocalEwaldState(state), real_space(state) {
        generateKVectors(state.box_length);
    }

    // @todo incomplete; mostly copied from faunus
    inline void generateKVectors(const vec3 &box_length) {
        auto inside_cutoff = [cutoff_squared = reciprocal_cutoff * reciprocal_cutoff](auto nx, auto ny, auto nz) {
            const auto r = nx * nx + ny * ny + nz * nz;
            return (r > 0 && r <= cutoff_squared);
        }; // lambda to determine if wave-vector is within spherical cut-off

        this->box_length = box_length;
        int number_of_k_vectors = std::pow(2 * reciprocal_cutoff + 1, 3) - 1;
        if (number_of_k_vectors > 0) {
            resize(number_of_k_vectors); // allocate maximum possible number of k-vectors
            number_of_k_vectors = 0;     // reset and count again...
            const vec3 two_pi_inverse_box_length = 2.0 * pi * box_length.cwiseInverse();
            for (int nx = -reciprocal_cutoff; nx < reciprocal_cutoff + 1; nx++) {
                for (int ny = -reciprocal_cutoff; ny < reciprocal_cutoff + 1; ny++) {
                    for (int nz = -reciprocal_cutoff; nz < reciprocal_cutoff + 1; nz++) {
                        if (inside_cutoff(nx, ny, nz)) {
                            k_vectors.col(number_of_k_vectors++) =
                                two_pi_inverse_box_length.cwiseProduct(vec3(nx, ny, nz));
                        }
                    }
                }
            }
        }
        resize(number_of_k_vectors); // shrink if needed due to above cutoff
        setAks();
        Q_mp.setZero();
    }

    /**
     * @brief Updates Q for a set of particles
     *
     * Calculated values are by default _added_ to `Q_mp`, but can be set to
     * subtract with `binary_op = std::minus<>()`. This can be useful for optimized
     * updates of changes to a subset of the particles.
     */
    template <class Positions, class Charges, class Dipoles, class BinaryOp = std::plus<>>
    void updateComplex(Positions &positions, Charges &charges, Dipoles &dipoles, BinaryOp binary_op = std::plus<>()) {
        for (int i = 0; i < k_vectors.cols(); i++) {
            Q_mp[i] = binary_op(Q_mp[i], calcQ(k_vectors.col(i), positions, charges, dipoles));
        }
    }

    /**
     * @brief Reciprocal-space energy
     */
    inline auto reciprocal_energy() {
        if constexpr (true) { // Eigen library syntax
            return 2.0 * pi / getVolume() * Q_mp.cwiseAbs2().cwiseProduct(Aks).sum();
        } else {
            double sum = 0.0;
            for (int i = 0; i < k_vectors.cols(); i++) {
                const auto absQ = std::abs(Q_mp[i]);
                sum += absQ * absQ * Aks[i];
            }
            return 2.0 * pi / getVolume() * sum;
        }
    }

    /**
     * @brief Reciprocal-space force
     * @param position Position of particle
     * @param charge Charge of particle
     * @param dipole_moment Dipole moment of particle
     */
    inline vec3 reciprocal_force(const vec3 &position, const double charge, const vec3 &dipole_moment) {
        vec3 sum = {0.0, 0.0, 0.0};
        for (int i = 0; i < k_vectors.cols(); i++) {
            const auto &k = k_vectors.col(i);
            const auto kr = k.dot(position); // 𝒌⋅𝒓
            const auto qmu = Tcomplex(-dipole_moment.dot(k), charge);
            const auto repart = Tcomplex(std::cos(kr), std::sin(kr)) * qmu * std::conj(Q_mp[i]);
            sum += std::real(repart) * k * Aks[i];
        }
        return -4.0 * pi / getVolume() * sum;
    }

    /**
     * @brief Reciprocal-space field
     * @param position Evaluation position
     */
    vec3 reciprocal_field(const vec3 &position) {
        vec3 field = {0.0, 0.0, 0.0};
        for (int i = 0; i < k_vectors.cols(); i++) {
            const auto kr = k_vectors.col(i).dot(position); // 𝒌⋅𝒓
            const auto repart = Tcomplex(-std::sin(kr), std::cos(kr)) * std::conj(Q_mp[i]);
            field += std::real(repart) * k_vectors.col(i) * Aks[i];
        }
        return -4.0 * pi / getVolume() * field;
    }

    /**
     * @brief Surface field-term
     * @param positions Positions of particles
     * @param charges Charges of particles
     * @param dipoles Dipole moments of particles
     */
    template <typename Positions, typename Charges, typename Dipoles>
    vec3 surface_field(const Positions &positions, const Charges &charges, const Dipoles &dipoles) const {
        auto total_dipole = dipoleMoment(positions, charges, dipoles);
        return -4.0 * pi / (2.0 * surface_dielectric_constant + 1.0) / getVolume() * total_dipole;
    }

    /**
     * @brief Surface energy-term
     * @param positions Positions of particles
     * @param charges Charges of particles
     * @param dipoles Dipole moments of particles
     */
    template <typename Positions, typename Charges, typename Dipoles>
    double surface_energy(const Positions &positions, const Charges &charges, const Dipoles &dipoles) const {
        auto total_dipole = dipoleMoment(positions, charges, dipoles);
        return 2.0 * pi / (2.0 * surface_dielectric_constant + 1.0) / getVolume() * total_dipole.squaredNorm();
    }

    /**
     * @brief Surface force-term
     * @param positions Positions of particles
     * @param charges Charges of particles
     * @param dipoles Dipole moments of particles
     * @param charge Charge of particle
     * @warning Only works for charges
     */
    template <typename Positions, typename Charges, typename Dipoles>
    vec3 surface_force(Positions &positions, Charges &charges, Dipoles &dipoles, const double charge) const {
        return charge * surface_field(positions, charges, dipoles);
    }
};

// -------------- Ewald real-space (using truncated Gaussian) ---------------

/**
 * @brief Ewald real-space scheme using a truncated Gaussian screening-function.
 */
class EwaldTruncated : public EnergyImplementation<EwaldTruncated> {
    double eta, eta2, eta3; //!< Reduced damping-parameter, and squared, and cubed
    // double zeta, zeta2, zeta3;             //!< Reduced inverse Debye-length, and squared, and cubed
    double surface_dielectric_constant; //!< Dielectric constant of the surrounding medium
    double F0;                          //!< 'scaling' of short-ranged function
    const double pi_sqrt = 2.0 * std::sqrt(std::atan(1.0));
    const double pi = 4.0 * std::atan(1.0);

  public:
    /**
     * @param cutoff distance cutoff
     * @param alpha damping-parameter
     */
    inline EwaldTruncated(double cutoff, double alpha, double surface_dielectric_constant = infinity,
                          [[maybe_unused]] double debye_length = infinity)
        : EnergyImplementation(Scheme::ewaldt, cutoff), surface_dielectric_constant(surface_dielectric_constant) {
        name = "EwaldT real-space";
        has_dipolar_selfenergy = true;
        doi = "XYZ";
        eta = alpha * cutoff;
        eta2 = eta * eta;
        eta3 = eta2 * eta;
        if (surface_dielectric_constant < 1.0) {
            surface_dielectric_constant = infinity;
        }
        F0 = 1.0 - std::erfc(eta) - 2.0 * eta / pi_sqrt * std::exp(-eta2);
        if (std::isinf(surface_dielectric_constant)) {
            T0 = 1.0;
        } else {
            T0 = 2.0 * (surface_dielectric_constant - 1.0) / (2.0 * surface_dielectric_constant + 1.0);
        }
        chi = -(1.0 - 4.0 * eta3 * std::exp(-eta2) / (3.0 * pi_sqrt * F0)) * cutoff_squared * pi / eta2;
        setSelfEnergyPrefactor({-eta / pi_sqrt * (1.0 - std::exp(-eta2)) / F0,
                                -eta3 * 2.0 / 3.0 / (std::erf(eta) * pi_sqrt - 2.0 * eta * std::exp(-eta2))});
        setSelfFieldPrefactor({0.0, 0.0}); // FIX
    }

    inline double short_range_function(double q) const override {
        return (std::erfc(eta * q) - std::erfc(eta) - (1.0 - q) * 2.0 * eta / pi_sqrt * std::exp(-eta2)) / F0;
    }
    inline double short_range_function_derivative(double q) const override {
        return -2.0 * eta * (std::exp(-eta2 * q * q) - std::exp(-eta2)) / pi_sqrt / F0;
    }
    inline double short_range_function_second_derivative(double q) const override {
        return 4.0 * eta3 * q * std::exp(-eta2 * q * q) / pi_sqrt / F0;
    }
    inline double short_range_function_third_derivative(double q) const override {
        return -8.0 * (eta2 * q * q - 0.5) * eta3 * std::exp(-eta2 * q * q) / pi_sqrt / F0;
    }

    /**
     * @brief Surface-term
     * @param positions Positions of particles
     * @param charges Charges of particles
     * @param dipoles Dipole moments of particles
     * @param volume Volume of unit-cell
     */
    inline double surface_energy(const std::vector<vec3> &positions, const std::vector<double> &charges,
                                 const std::vector<vec3> &dipoles, double volume) const {
        assert(positions.size() == charges.size());
        assert(positions.size() == dipoles.size());
        vec3 pos_times_charge_sum = {0.0, 0.0, 0.0};
        vec3 dipole_sum = {0.0, 0.0, 0.0};
        for (size_t i = 0; i < positions.size(); i++) {
            pos_times_charge_sum += positions[i] * charges[i];
            dipole_sum += dipoles[i];
        }
        const auto sqDipoles = pos_times_charge_sum.dot(pos_times_charge_sum) +
                               2.0 * pos_times_charge_sum.dot(dipole_sum) + dipole_sum.dot(dipole_sum);
        return (2.0 * pi / (2.0 * surface_dielectric_constant + 1.0) / volume * sqDipoles);
    }

#ifdef NLOHMANN_JSON_HPP
    inline EwaldTruncated(const nlohmann::json &j)
        : EwaldTruncated(j.at("cutoff").get<double>(), j.at("alpha").get<double>(), j.value("epss", infinity),
                         j.value("debyelength", infinity)) {}

  private:
    inline void _to_json(nlohmann::json &j) const override {
        j = {{"alpha", eta / cutoff}};
        if (std::isinf(surface_dielectric_constant)) {
            j["epss"] = "inf";
        } else {
            j["epss"] = surface_dielectric_constant;
        }
    }
#endif
};
} // namespace CoulombGalore
