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

#include "ewald.hpp"
#include "ewald_truncated.hpp"
#include "fanourgakis.hpp"
#include "fennell.hpp"
#include "plain.hpp"
#include "poisson.hpp"
#include "qpotential.hpp"
#include "reactionfield.hpp"
#include "splined.hpp"
#include "wolf.hpp"
#include "zahn.hpp"
#include "zerodipole.hpp"

namespace CoulombGalore {
#ifdef NLOHMANN_JSON_HPP
inline std::shared_ptr<SchemeBase> createScheme(const nlohmann::json &j) {
    const std::map<std::string, Scheme> m = {{"plain", Scheme::plain},
                                             {"qpotential", Scheme::qpotential},
                                             {"wolf", Scheme::wolf},
                                             {"poisson", Scheme::poisson},
                                             {"reactionfield", Scheme::reactionfield},
                                             {"spline", Scheme::spline},
                                             {"fanourgakis", Scheme::fanourgakis},
                                             {"fennell", Scheme::fennell},
                                             {"zahn", Scheme::zahn},
                                             {"zerodipole", Scheme::zerodipole},
                                             {"ewald", Scheme::ewald},
                                             {"ewaldt", Scheme::ewaldt}}; // map string keyword to scheme type

    std::string name = j.at("type").get<std::string>();
    auto it = m.find(name);
    if (it == m.end()) {
        throw std::runtime_error("unknown coulomb scheme " + name);
    }

    std::shared_ptr<SchemeBase> scheme;
    switch (it->second) {
    case Scheme::plain:
        scheme = std::make_shared<Plain>(j);
        break;
    case Scheme::wolf:
        scheme = std::make_shared<Wolf>(j);
        break;
    case Scheme::zahn:
        scheme = std::make_shared<Zahn>(j);
        break;
    case Scheme::fennell:
        scheme = std::make_shared<Fennell>(j);
        break;
    case Scheme::zerodipole:
        scheme = std::make_shared<ZeroDipole>(j);
        break;
    case Scheme::fanourgakis:
        scheme = std::make_shared<Fanourgakis>(j);
        break;
    case Scheme::qpotential5:
        scheme = std::make_shared<qPotentialFixedOrder<5>>(j);
        break;
    case Scheme::qpotential:
        scheme = std::make_shared<qPotential>(j);
        break;
    case Scheme::ewald:
        scheme = std::make_shared<Ewald>(j);
        break;
    case Scheme::ewaldt:
        scheme = std::make_shared<EwaldTruncated>(j);
        break;
    case Scheme::poisson:
        scheme = std::make_shared<Poisson>(j);
        break;
    case Scheme::reactionfield:
        scheme = std::make_shared<ReactionField>(j);
        break;
    default:
        break;
    }
    return scheme;
}
#endif

} // namespace CoulombGalore
