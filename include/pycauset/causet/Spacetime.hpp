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

    // Volume of the spacetime region
    virtual double volume() const = 0;
};

class MinkowskiDiamond : public CausalSpacetime {
    int dim;
public:
    MinkowskiDiamond(int dimension) : dim(dimension) {}
    
    int dimension() const override { return dim; }
    
    double volume() const override {
        // Unit diamond in lightcone coordinates [0, 1]^dim
        // Volume is 1.0 in these coordinates.
        return 1.0;
    }
    
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
        // For higher dimensions, we need a proper implementation.
        // Assuming product of lightcone intervals for now (hyper-diamond)
        for(int i=0; i<dim; ++i) {
            if (u[i] >= v[i]) return false;
        }
        return true;
    }
};

class MinkowskiCylinder : public CausalSpacetime {
    int dim;
    double height;
    double circumference;
public:
    MinkowskiCylinder(int dimension, double height, double circumference) 
        : dim(dimension), height(height), circumference(circumference) {
        if (dim != 2) {
            // Only 2D supported for now
            throw std::runtime_error("MinkowskiCylinder only supports dimension 2 currently.");
        }
    }
    
    int dimension() const override { return dim; }

    double get_height() const { return height; }
    double get_circumference() const { return circumference; }
    
    double volume() const override {
        return height * circumference;
    }
    
    std::vector<double> generate_point(std::mt19937_64& rng) const override {
        // Coordinates: (t, x)
        // t in [0, height]
        // x in [0, circumference]
        std::uniform_real_distribution<double> dist_t(0.0, height);
        std::uniform_real_distribution<double> dist_x(0.0, circumference);
        
        return {dist_t(rng), dist_x(rng)};
    }
    
    bool causality(const std::vector<double>& u, const std::vector<double>& v) const override {
        // u = (t1, x1), v = (t2, x2)
        double t1 = u[0];
        double x1 = u[1];
        double t2 = v[0];
        double x2 = v[1];
        
        if (t1 >= t2) return false;
        
        double dt = t2 - t1;
        double dx = std::abs(x2 - x1);
        
        // Shortest distance on the circle
        dx = std::min(dx, circumference - dx);
        
        // Lightcone condition: dt > dx
        return dt > dx;
    }
};

class MinkowskiBox : public CausalSpacetime {
    int dim;
    double time_extent;
    double space_extent;
public:
    MinkowskiBox(int dimension, double t_len, double x_len) 
        : dim(dimension), time_extent(t_len), space_extent(x_len) {}
    
    int dimension() const override { return dim; }
    
    double get_time_extent() const { return time_extent; }
    double get_space_extent() const { return space_extent; }

    double volume() const override {
        // Volume = T * L^(d-1)
        return time_extent * std::pow(space_extent, dim - 1);
    }
    
    std::vector<double> generate_point(std::mt19937_64& rng) const override {
        // Coordinates: (t, x, y, ...)
        std::uniform_real_distribution<double> dist_t(0.0, time_extent);
        std::uniform_real_distribution<double> dist_x(0.0, space_extent);
        
        std::vector<double> p(dim);
        p[0] = dist_t(rng);
        for(int i=1; i<dim; ++i) {
            p[i] = dist_x(rng);
        }
        return p;
    }
    
    bool causality(const std::vector<double>& u, const std::vector<double>& v) const override {
        // u = (t1, x1...), v = (t2, x2...)
        if (u[0] >= v[0]) return false;
        
        double dt = v[0] - u[0];
        double dx_sq = 0.0;
        for(int i=1; i<dim; ++i) {
            double d = v[i] - u[i];
            dx_sq += d*d;
        }
        
        // dt^2 > dx^2
        return (dt*dt) > dx_sq;
    }
};

}

#endif // PYCAUSET_SPACETIME_HPP
