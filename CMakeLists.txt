cmake_minimum_required(VERSION 3.15)
project(pycauset VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---

# 1. Python (Required for the extension)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

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

# 3. OpenMP (Optional but recommended)
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
)

# --- Extension Module ---
# Note: The module name must match what is defined in PYBIND11_MODULE in bindings.cpp (which is "_pycauset")
# Touch
python_add_library(_pycauset MODULE src/bindings.cpp ${PYCAUSET_SOURCES})

# Include directories
target_include_directories(_pycauset PRIVATE include)

# Link libraries
target_link_libraries(_pycauset PRIVATE pybind11::module)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_pycauset PRIVATE OpenMP::OpenMP_CXX)
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

    add_executable(causal_tests tests/test_main.cpp ${PYCAUSET_SOURCES})
    # Remove bindings.cpp from tests if it contains the main module definition which might conflict
    # Actually bindings.cpp has PYBIND11_MODULE, which is fine in a library but maybe not in an executable if it has main?
    # test_main.cpp has main(). bindings.cpp does not have main().
    
    target_include_directories(causal_tests PRIVATE include)
    target_link_libraries(causal_tests PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
    
    add_test(NAME CausalTests COMMAND causal_tests)
endif()
