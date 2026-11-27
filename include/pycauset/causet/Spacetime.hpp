#pragma once
// Spacetimes the native sprinkler can sample (R1: the Minkowski causal diamond in any
// dimension; the rest of the spacetime library is parked in experimental/physics).

#include <cmath>
#include <numbers>
#include <random>
#include <stdexcept>
#include <vector>

namespace pycauset {

class CausalSpacetime {
public:
    virtual ~CausalSpacetime() = default;
    virtual int dimension() const = 0;
    // One point, uniformly distributed in the region. Coordinate 0 is time.
    virtual std::vector<double> generate_point(std::mt19937_64& rng) const = 0;
    // True when u is in the causal past of v (u < v).
    virtual bool causality(const std::vector<double>& u, const std::vector<double>& v) const = 0;
    virtual double volume() const = 0;
};

// The causal diamond I+(p) ∩ I-(q) in d-dimensional Minkowski space, p = (0, 0), q = (1, 0),
// in coordinates (t, x_1, ..., x_{d-1}), c = 1: the points with |x| < min(t, 1 - t).
// Sampling: uniform in the bounding box t in [0, 1], x_i in [-1/2, 1/2], rejecting points
// outside the diamond (exactly uniform; acceptance 2 V_{d-1} / (d 2^d): 1/2 in d = 2, ~0.13
// in d = 4). Causality: u < v iff (t_v - t_u)^2 > |x_v - x_u|^2 and t_v > t_u.
class MinkowskiDiamond : public CausalSpacetime {
public:
    explicit MinkowskiDiamond(int dimension) : dim_(dimension) {
        if (dimension < 2) throw std::invalid_argument("MinkowskiDiamond: dimension must be >= 2");
    }
    int dimension() const override { return dim_; }

    // Volume of the diamond: 2 * V_{d-1} * (1/2)^d / d, V_k = pi^(k/2) / Gamma(k/2 + 1) the unit
    // k-ball (1/2 in d = 2).
    double volume() const override {
        const double k = dim_ - 1;
        const double ball = std::pow(std::numbers::pi, k / 2.0) / std::tgamma(k / 2.0 + 1.0);
        return 2.0 * ball * std::pow(0.5, dim_) / dim_;
    }

    std::vector<double> generate_point(std::mt19937_64& rng) const override {
        std::uniform_real_distribution<double> t_dist(0.0, 1.0), x_dist(-0.5, 0.5);
        std::vector<double> p(dim_);
        for (;;) {
            p[0] = t_dist(rng);
            double r2 = 0.0;
            for (int i = 1; i < dim_; ++i) {
                p[i] = x_dist(rng);
                r2 += p[i] * p[i];
            }
            const double h = std::min(p[0], 1.0 - p[0]);
            if (r2 < h * h) return p;
        }
    }

    bool causality(const std::vector<double>& u, const std::vector<double>& v) const override {
        const double dt = v[0] - u[0];
        if (dt <= 0.0) return false;
        double dx2 = 0.0;
        for (int i = 1; i < dim_; ++i) {
            const double d = v[i] - u[i];
            dx2 += d * d;
        }
        return dt * dt > dx2;
    }

private:
    int dim_;
};

}  // namespace pycauset
