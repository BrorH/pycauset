cmake_minimum_required(VERSION 3.15)
project(CausalPlusPlus VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Pybind11 for Python bindings
FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.12.0.zip
)
FetchContent_MakeAvailable(pybind11)

find_package(OpenMP)

include_directories(include)

add_library(causal_lib 
    src/MemoryMapper.cpp
    src/MatrixBase.cpp
    src/TriangularMatrix.cpp
    src/CausalMatrix.cpp
    src/IntegerMatrix.cpp
    src/KMatrix.cpp
    src/TriangularFloatMatrix.cpp
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(causal_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# Python Module
pybind11_add_module(pycauset src/bindings.cpp)
if (COMMAND pybind11_enable_stable_abi)
  pybind11_enable_stable_abi(pycauset 3.8)
endif()
target_link_libraries(pycauset PRIVATE causal_lib)

add_executable(causal_main src/main.cpp)
target_link_libraries(causal_main PRIVATE causal_lib)

enable_testing()

add_executable(causal_tests tests/test_main.cpp)
target_link_libraries(causal_tests PRIVATE causal_lib GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(causal_tests)
