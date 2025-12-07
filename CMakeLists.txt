cmake_minimum_required(VERSION 3.15)
project(pycauset VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---

# 1. Python (Required for the extension)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# 2. Pybind11 (Required)
# If built via pip/scikit-build-core, pybind11 is provided.
# If built manually, we might need to fetch it.
find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      pybind11
      URL https://github.com/pybind/pybind11/archive/refs/tags/v2.12.0.zip
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# 3. Eigen (Required for advanced solvers)
include(FetchContent)
FetchContent_Declare(
  Eigen3
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
)
FetchContent_MakeAvailable(Eigen3)

# 4. OpenMP (Optional but recommended)
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND AND APPLE)
    # Try to find OpenMP in Homebrew paths on macOS
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "omp")
    
    # Check standard Homebrew locations
    file(GLOB OPENMP_DIRS "/opt/homebrew/opt/libomp" "/usr/local/opt/libomp")
    foreach(DIR ${OPENMP_DIRS})
        if(EXISTS "${DIR}/include/omp.h")
            include_directories("${DIR}/include")
            link_directories("${DIR}/lib")
            set(OpenMP_CXX_FOUND TRUE)
            message(STATUS "Found OpenMP via Homebrew at ${DIR}")
            break()
        endif()
    endforeach()
endif()

# --- Source Files ---
set(PYCAUSET_SOURCES
    src/MemoryMapper.cpp
    src/PersistentObject.cpp
    src/StoragePaths.cpp
    src/MatrixBase.cpp
    src/VectorBase.cpp
    src/TriangularMatrix.cpp
    src/TriangularBitMatrix.cpp
    src/DenseBitMatrix.cpp
    src/DenseVector.cpp
    src/MatrixOperations.cpp
    src/VectorOperations.cpp
    src/MatrixFactory.cpp
    src/VectorFactory.cpp
    src/Sprinkler.cpp
    src/ComplexVector.cpp
    src/Eigen.cpp
    src/ParallelUtils.cpp
    src/ComputeContext.cpp
    src/CpuDevice.cpp
    src/CpuSolver.cpp
    src/SystemUtils.cpp
    src/AutoSolver.cpp
)

# --- Extension Module ---
# Note: The module name must match what is defined in PYBIND11_MODULE in bindings.cpp (which is "_pycauset")
# Touch
python_add_library(_pycauset SHARED src/bindings.cpp ${PYCAUSET_SOURCES})

# Include directories
target_include_directories(_pycauset PRIVATE include)

# Link libraries
target_link_libraries(_pycauset PRIVATE pybind11::module Eigen3::Eigen)

if(MSVC)
    target_compile_options(_pycauset PRIVATE /bigobj)
endif()

# Export symbols for plugins
set_target_properties(_pycauset PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_pycauset PRIVATE OpenMP::OpenMP_CXX)
endif()

# --- CUDA Acceleration (Optional) ---
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    # Target the native architecture of the build machine (supports Pascal 6.1 on CUDA 12)
    set(CMAKE_CUDA_ARCHITECTURES "native")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    message(STATUS "CUDA found. Building pycauset_cuda accelerator.")
    
    add_library(pycauset_cuda SHARED
        src/accelerators/cuda/CudaDevice.cu
        src/accelerators/cuda/CudaSolver.cu
    )
    
    target_include_directories(pycauset_cuda PRIVATE include)
    
    # Link against CUDA libraries
    target_link_libraries(pycauset_cuda PRIVATE 
        CUDA::cudart 
        CUDA::cublas 
        CUDA::cusolver
        _pycauset
    )
    
    # Set properties
    set_target_properties(pycauset_cuda PROPERTIES 
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION ON
    )
    
    install(TARGETS pycauset_cuda DESTINATION pycauset)
else()
    message(STATUS "CUDA not found. Skipping pycauset_cuda accelerator.")
endif()

# --- Installation ---
# This installs the compiled extension into the python package directory
install(TARGETS _pycauset DESTINATION pycauset)

# --- Testing (Only if explicitly requested or not a pip build) ---
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(causal_tests tests/test_main.cpp tests/test_parallel.cpp tests/test_gpu.cpp ${PYCAUSET_SOURCES})
    # Remove bindings.cpp from tests if it contains the main module definition which might conflict
    # Actually bindings.cpp has PYBIND11_MODULE, which is fine in a library but maybe not in an executable if it has main?
    # test_main.cpp has main(). bindings.cpp does not have main().
    
    target_include_directories(causal_tests PRIVATE include)
    target_link_libraries(causal_tests PRIVATE GTest::gtest_main Eigen3::Eigen)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(causal_tests PRIVATE OpenMP::OpenMP_CXX)
    endif()
    
    # Link CUDA if available for GPU tests
    if(CUDAToolkit_FOUND)
        target_link_libraries(causal_tests PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver pycauset_cuda)
        target_compile_definitions(causal_tests PRIVATE ENABLE_CUDA)
    endif()
    
    add_test(NAME CausalTests COMMAND causal_tests)

    # Type System Tests
    add_executable(test_types tests/test_types.cpp ${PYCAUSET_SOURCES})
    target_include_directories(test_types PRIVATE include)
    target_link_libraries(test_types PRIVATE GTest::gtest_main Eigen3::Eigen)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_types PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(CUDAToolkit_FOUND)
        target_link_libraries(test_types PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver pycauset_cuda)
        target_compile_definitions(test_types PRIVATE ENABLE_CUDA)
    endif()
    add_test(NAME TypeTests COMMAND test_types)

    # Benchmark Executable
    add_executable(benchmark_tests tests/benchmark_parallel.cpp ${PYCAUSET_SOURCES})
    target_include_directories(benchmark_tests PRIVATE include)
    target_link_libraries(benchmark_tests PRIVATE Eigen3::Eigen)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(benchmark_tests PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # Optimization Benchmark
    add_executable(benchmark_optimizations tests/benchmark_optimizations.cpp ${PYCAUSET_SOURCES})
    target_include_directories(benchmark_optimizations PRIVATE include)
    target_link_libraries(benchmark_optimizations PRIVATE Eigen3::Eigen)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(benchmark_optimizations PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(CUDAToolkit_FOUND)
        target_link_libraries(benchmark_optimizations PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver pycauset_cuda)
        target_compile_definitions(benchmark_optimizations PRIVATE ENABLE_CUDA)
    endif()

    # GPU Benchmark Executable
    if(CUDAToolkit_FOUND)
        add_executable(benchmark_gpu tests/benchmark_gpu.cpp ${PYCAUSET_SOURCES})
        target_include_directories(benchmark_gpu PRIVATE include)
        target_link_libraries(benchmark_gpu PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver pycauset_cuda Eigen3::Eigen)
        target_compile_definitions(benchmark_gpu PRIVATE ENABLE_CUDA)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(benchmark_gpu PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endif()

    # GPU Float16 Tests
    if(CUDAToolkit_FOUND)
        add_executable(test_gpu_float16 tests/test_gpu_float16.cpp ${PYCAUSET_SOURCES})
        target_include_directories(test_gpu_float16 PRIVATE include src/accelerators/cuda)
        target_link_libraries(test_gpu_float16 PRIVATE GTest::gtest_main CUDA::cudart CUDA::cublas CUDA::cusolver pycauset_cuda Eigen3::Eigen)
        target_compile_definitions(test_gpu_float16 PRIVATE ENABLE_CUDA)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(test_gpu_float16 PRIVATE OpenMP::OpenMP_CXX)
        endif()
        add_test(NAME GpuFloat16Tests COMMAND test_gpu_float16)
    endif()

    # Pinned Memory Tests
    add_executable(test_pinned_memory tests/test_pinned_memory.cpp ${PYCAUSET_SOURCES})
    target_include_directories(test_pinned_memory PRIVATE include src/accelerators/cuda)
    target_link_libraries(test_pinned_memory PRIVATE GTest::gtest_main Eigen3::Eigen)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_pinned_memory PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(CUDAToolkit_FOUND)
        target_link_libraries(test_pinned_memory PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver pycauset_cuda)
        target_compile_definitions(test_pinned_memory PRIVATE ENABLE_CUDA)
    endif()
    add_test(NAME PinnedMemoryTests COMMAND test_pinned_memory)

    # Pinned Memory Extended Tests
    add_executable(test_pinned_memory_extended tests/test_pinned_memory_extended.cpp ${PYCAUSET_SOURCES})
    target_include_directories(test_pinned_memory_extended PRIVATE include src/accelerators/cuda)
    target_link_libraries(test_pinned_memory_extended PRIVATE GTest::gtest_main Eigen3::Eigen)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_pinned_memory_extended PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(CUDAToolkit_FOUND)
        target_link_libraries(test_pinned_memory_extended PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver pycauset_cuda)
        target_compile_definitions(test_pinned_memory_extended PRIVATE ENABLE_CUDA)
    endif()
    add_test(NAME PinnedMemoryExtendedTests COMMAND test_pinned_memory_extended)

    # Solver Tests
    add_executable(test_solvers tests/test_solvers.cpp ${PYCAUSET_SOURCES})
    target_include_directories(test_solvers PRIVATE include src/accelerators/cuda)
    target_link_libraries(test_solvers PRIVATE GTest::gtest_main Eigen3::Eigen)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_solvers PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(CUDAToolkit_FOUND)
        target_link_libraries(test_solvers PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver pycauset_cuda)
        target_compile_definitions(test_solvers PRIVATE ENABLE_CUDA)
    endif()
    add_test(NAME SolverTests COMMAND test_solvers)

    # Extensive Solver Benchmark
    add_executable(benchmark_solvers_extensive tests/benchmark_solvers_extensive.cpp ${PYCAUSET_SOURCES})
    target_include_directories(benchmark_solvers_extensive PRIVATE include src/accelerators/cuda)
    target_link_libraries(benchmark_solvers_extensive PRIVATE Eigen3::Eigen)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(benchmark_solvers_extensive PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(CUDAToolkit_FOUND)
        target_link_libraries(benchmark_solvers_extensive PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver pycauset_cuda)
        target_compile_definitions(benchmark_solvers_extensive PRIVATE ENABLE_CUDA)
    endif()
endif()
