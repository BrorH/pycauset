#ifndef PYCAUSET_SPACETIME_HPP
#define PYCAUSET_SPACETIME_HPP

#pragma message("Spacetime.hpp included")
#include <vector>
#include <random>
#include <cmath>
#include <memory>

namespace pycauset {

class CausalSpacetime {
public:
    virtual ~CausalSpacetime() = default;
    
    // Dimension of the spacetime (e.g., 2 for 1+1D)
    virtual int dimension() const = 0;
    
    // Generate a single point in the spacetime using the provided RNG
    virtual std::vector<double> generate_point(std::mt19937_64& rng) const = 0;
    
    // Check if u is causally in the past of v (u < v)
    virtual bool causality(const std::vector<double>& u, const std::vector<double>& v) const = 0;
};

class MinkowskiDiamond : public CausalSpacetime {
    int dim;
public:
    MinkowskiDiamond(int dimension) : dim(dimension) {}
    
    int dimension() const override { return dim; }
    
    std::vector<double> generate_point(std::mt19937_64& rng) const override {
        // For 2D (1+1): Lightcone coordinates u, v in [0, 1].
        // This represents a causal diamond.
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::vector<double> p(dim);
        for(int i=0; i<dim; ++i) {
            p[i] = dist(rng);
        }
        return p;
    }
    
    bool causality(const std::vector<double>& u, const std::vector<double>& v) const override {
        if (dim == 2) {
            // In lightcone coordinates for a diamond:
            // u < v iff u_0 < v_0 AND u_1 < v_1
            return (u[0] < v[0]) && (u[1] < v[1]);
        }
        return false; 
    }
};

}

#endif // PYCAUSET_SPACETIME_HPP
