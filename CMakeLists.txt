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
    src/core/MemoryMapper.cpp
    src/core/MemoryGovernor.cpp
    src/core/IOAccelerator.cpp
    src/core/PersistentObject.cpp
    src/core/StorageUtils.cpp
    src/core/SystemUtils.cpp
    src/core/ParallelUtils.cpp
    src/matrix/MatrixBase.cpp
    src/matrix/DenseBitMatrix.cpp
    src/matrix/TriangularMatrix.cpp
    src/matrix/TriangularBitMatrix.cpp
    src/core/ObjectFactory.cpp
    src/vector/VectorBase.cpp
    src/vector/DenseVector.cpp
    src/vector/ComplexVector.cpp
    src/compute/ComputeContext.cpp
    src/compute/AutoSolver.cpp
    src/compute/cpu/CpuDevice.cpp
    src/compute/cpu/CpuSolver.cpp
    src/math/Eigen.cpp
    src/math/LinearAlgebra.cpp
    src/causet/Sprinkler.cpp
)

# Create a shared library for the core functionality to avoid recompiling for every test
add_library(pycauset_core SHARED ${PYCAUSET_SOURCES})
target_include_directories(pycauset_core PUBLIC include)
target_link_libraries(pycauset_core PUBLIC Eigen3::Eigen)
set_property(TARGET pycauset_core PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(pycauset_core PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(OpenMP_CXX_FOUND)
    target_link_libraries(pycauset_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(pycauset_core PRIVATE /bigobj)
endif()

# --- Extension Module ---
# Note: The module name must match what is defined in PYBIND11_MODULE in bindings.cpp (which is "_pycauset")
# Touch
python_add_library(_pycauset SHARED src/bindings.cpp)

# Include directories
target_include_directories(_pycauset PRIVATE include)

# Link libraries
target_link_libraries(_pycauset PRIVATE pybind11::module pycauset_core)

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
        pycauset_core
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

    # Memory Governor Tests
    add_executable(test_memory_governor tests/test_memory_governor.cpp)
    target_include_directories(test_memory_governor PRIVATE include)
    target_link_libraries(test_memory_governor PRIVATE GTest::gtest_main pycauset_core)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_memory_governor PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_test(NAME MemoryGovernorTests COMMAND test_memory_governor)

    # IO Accelerator Tests
    add_executable(test_io_accelerator tests/test_io_accelerator.cpp)
    target_include_directories(test_io_accelerator PRIVATE include)
    target_link_libraries(test_io_accelerator PRIVATE GTest::gtest_main pycauset_core)
    add_test(NAME IOAcceleratorTests COMMAND test_io_accelerator)

    # CoW Tests
    add_executable(test_cow tests/test_cow.cpp)
    target_include_directories(test_cow PRIVATE include)
    target_link_libraries(test_cow PRIVATE GTest::gtest_main pycauset_core)
    add_test(NAME CoWTests COMMAND test_cow)

    # Other tests disabled for now as source files are missing in this context
    # add_executable(causal_tests tests/test_main.cpp tests/test_parallel.cpp tests/test_gpu.cpp)
    # ...

endif()
