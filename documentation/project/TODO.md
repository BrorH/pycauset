# Project Roadmap & TODO

This document outlines the development roadmap for `pycauset`, tracking completed features, active tasks, and future goals.

## 🚀 High Priority

- [ ] **Large Scale Simulation**: Calculate a 100GB propagator matrix $K$.
- [ ] **Further I/O Optimization**: The read/write speed to disk is currently the main bottleneck
- [ ] **Pauli-Jordan Function**: Implement the $i\Delta$ function derived from the propagator.
- [ ] **GPU Acceleration**: Investigate CUDA/OpenCL support for matrix operations.
- [ ] **User-Defined Spacetimes**: Allow users to define custom spacetime geometries and boundaries in Python.


## 🌌 Physics & Spacetime

- [x] **Propagator Calculation**: Implemented $K = \Phi (I - b\Phi)^{-1}$ (Eq. 3.5 in Johnston).
- [x] **Spacetime Shapes**:
    - [x] Minkowski Diamond (Null boundaries)
    - [x] Minkowski Cylinder (Periodic boundaries)
    - [x] Minkowski Box (Hard wall boundaries)
- [x] **Field Theory**: Corrected coefficient calculation ($a, b$) based on dimension and mass in `ScalarField`.
- [ ] **Curved Spacetimes**: Implement Schwarzschild or de Sitter spacetimes.
- [ ] **Eigenvector Analysis**: Study base changes and eigenvector properties of the causal matrix.

## 🛠 Core Functionality

- [x] **Matrix Operations**: Addition, subtraction, multiplication, inversion.
- [x] **Storage Management**:
    - [x] Binary file format (`.pycauset`).
    - [x] Automatic temporary file cleanup.
    - [x] Configurable storage paths (including external drives).
    - [x] RAM-backed mapping for small objects.
- [x] **Data Types**:
    - [x] Dense and Triangular matrices.
    - [x] Vectors.
    - [x] Complex number support.
    - [x] `dtype` specification on creation.
- [x] **Numpy Compatibility**: Seamless conversion to/from numpy arrays.
- [x] **Identity Matrix**: `pycauset.I` with automatic sizing.
- [x] Create VectorFactory
	- [x] move "add_vectors" from "MatrixOperations.cpp"
## 📊 Visualization & Analysis

- [x] **Causet Visualization**: 2D and 3D plotting using Plotly.
- [x] **Coordinate Transforms**: Spacetime-aware coordinate transformations for visualization.
- [x] **Matrix Inspection**: Printable info and matrix content.

## 📚 Documentation

- [x] **Structure**: Reorganized into Guides, API, Internals, and Project sections.
- [x] **Hosting**: Setup for GitHub Pages / MkDocs.
- [ ] **Environment Variables**: Document `$PYCAUSET_STORAGE_DIR`.
- [ ] **Tutorials**: Add more physics-focused tutorials (e.g., scalar field propagation).
- [ ] vector cross product

## 🐛 Known Issues / Maintenance

- [ ] **Lifecycle Management**: Clarify `CausalSet` vs `CausalMatrix` persistence guarantees.
- [ ] **Legacy Cleanup**: Remove or update deprecated parameters (e.g., `populate=True` in `CausalMatrix`).

