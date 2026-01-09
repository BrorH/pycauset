cmake_minimum_required(VERSION 3.24)

# Set policies to silence warnings about timestamps and other modern features
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD) # Allow FindCUDA for now if needed, though we use CUDAToolkit
endif()
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 OLD) # Allow FindBoost for now
endif()
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD) # Allow FetchContent_Populate
endif()

project(pycauset VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_UNITY_BUILD OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Optimization: Unity Build (Jumbo Build) ---
# Merges source files into larger batches to reduce compiler overhead and link times.
# This is often the single most effective way to speed up C++ builds.
set(CMAKE_UNITY_BUILD OFF)

# --- Compiler Options ---
if(MSVC)
    # Define _CRT_SECURE_NO_WARNINGS to silence getenv/sprintf warnings globally
    # Define NOMINMAX to prevent Windows.h from defining min/max macros that conflict with std::min/std::max
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)

    # /MP: Multi-processor compilation (CXX Only - NVCC doesn't like it directly)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/MP>)

    # /wd4251: Suppress "needs to have dll-interface" warning
    # /wd4996: Suppress "deprecated/unsafe" warnings (getenv, etc.)
    # /wd4244: Suppress "conversion from 'type1' to 'type2', possible loss of data"
    # /wd4267: Suppress "conversion from 'size_t' to 'type', possible loss of data"
    # /wd4018: Suppress "signed/unsigned mismatch"
    # /wd4101: Suppress "unreferenced local variable"
    # /wd4804: Suppress "unsafe use of type 'bool' in operation"
    # Note: We do NOT add /W3 manually as CMake adds it by default, and adding it again can cause D9025 warnings if targets override it.
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/wd4251>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4996>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4244>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4267>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4018>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4101>
        $<$<COMPILE_LANGUAGE:CXX>:/wd4804>
    )
    
    # Disable LTO in Debug/RelWithDebInfo to speed up linking
    # In Release, we keep it default (usually off unless requested)
endif()

if(NOT MSVC)
    # Avoid -march=native for distribution wheels to ensure compatibility
    # add_compile_options(-march=native)
endif()

# --- CUDA Compiler Options ---
# (Moved to target definition to ensure correct application)

# --- RPATH Settings ---
# Ensure installed binaries can find adjacent shared libraries (libpycauset_core.so)
if(UNIX)
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

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
# Disable Eigen documentation and tests to silence build warnings
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)

include(FetchContent)
set(BUILD_TESTING OFF CACHE BOOL "Disable testing for dependencies")
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen testing")
set(EIGEN_BUILD_DOC OFF CACHE BOOL "Disable Eigen docs")
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "Disable Eigen pkgconfig")

FetchContent_Declare(
  Eigen3
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
)
FetchContent_MakeAvailable(Eigen3)

# 3.5 OpenBLAS (Required for high-performance matrix ops)
# Try to find it on the system first
message(STATUS "--- Debugging OpenBLAS Detection Start ---")
find_package(OpenBLAS)

if(OpenBLAS_FOUND)
   message(STATUS "Standard FindOpenBLAS succeeded.")
   if(TARGET OpenBLAS::OpenBLAS)
      message(STATUS "Target OpenBLAS::OpenBLAS exists.")
   else()
      message(STATUS "Target OpenBLAS::OpenBLAS DOES NOT EXIST despite Found=TRUE. Attempting to fix...")
      set(OpenBLAS_FOUND FALSE) # Force retry
   endif()
endif()

# Fallback: PkgConfig (Common on Linux)
if(NOT OpenBLAS_FOUND)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(PC_OPENBLAS openblas) # Removing QUIET to see why it fails
        if(PC_OPENBLAS_FOUND)
            message(STATUS "Found OpenBLAS via PkgConfig")
            if(NOT TARGET OpenBLAS::OpenBLAS)
                add_library(OpenBLAS::OpenBLAS INTERFACE IMPORTED)
                set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "${PC_OPENBLAS_INCLUDE_DIRS}"
                    INTERFACE_LINK_LIBRARIES "${PC_OPENBLAS_LINK_LIBRARIES}"
                    INTERFACE_LINK_DIRECTORIES "${PC_OPENBLAS_LIBRARY_DIRS}"
                    INTERFACE_COMPILE_OPTIONS "${PC_OPENBLAS_CFLAGS_OTHER}"
                )
            endif()
            set(OpenBLAS_FOUND TRUE)
        endif()
    endif()
endif()

# Fallback: Manual find_library (Last Resort)
if(NOT OpenBLAS_FOUND)
    message(STATUS "Attempting manual find_library...")
    
    # Reset variables to ensure fresh search
    unset(OpenBLAS_LIB CACHE)
    unset(OpenBLAS_INC CACHE)

    find_library(OpenBLAS_LIB NAMES openblas
        PATHS 
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
        /usr/lib
    )
    
    find_path(OpenBLAS_INC NAMES cblas.h
        PATHS 
        /usr/include/x86_64-linux-gnu
        /usr/include/openblas
        /usr/include
        /usr/local/include
    )
    
    message(STATUS "Manual find result: LIB=${OpenBLAS_LIB} INC=${OpenBLAS_INC}")

    if(OpenBLAS_LIB)
        if(NOT TARGET OpenBLAS::OpenBLAS)
            add_library(OpenBLAS::OpenBLAS UNKNOWN IMPORTED)
            set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
                IMPORTED_LOCATION "${OpenBLAS_LIB}"
            )
            if(OpenBLAS_INC)
                set_target_properties(OpenBLAS::OpenBLAS PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INC}")
            endif()
        endif()
        set(OpenBLAS_FOUND TRUE)
    endif()
endif()

if(NOT OpenBLAS_FOUND)
    message(STATUS "OpenBLAS not found on system. Downloading...")
    
    if(WIN32)
        # On Windows, download pre-built binaries to avoid Fortran requirement
        message(STATUS "Downloading OpenBLAS binaries for Windows (v0.3.26)...")
        FetchContent_Declare(
            OpenBLAS_Binaries
            URL https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.26/OpenBLAS-0.3.26-x64.zip
        )
        FetchContent_GetProperties(OpenBLAS_Binaries)
        if(NOT openblas_binaries_POPULATED)
            FetchContent_Populate(OpenBLAS_Binaries)
        endif()
        
        # Define the target manually
        if(NOT TARGET OpenBLAS::OpenBLAS)
            add_library(OpenBLAS::OpenBLAS SHARED IMPORTED)
            set(PYCAUSET_OPENBLAS_DLL "${openblas_binaries_SOURCE_DIR}/bin/libopenblas.dll")
            set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${openblas_binaries_SOURCE_DIR}/include"
                IMPORTED_IMPLIB "${openblas_binaries_SOURCE_DIR}/lib/libopenblas.lib"
                IMPORTED_LOCATION "${PYCAUSET_OPENBLAS_DLL}"
            )
            message(STATUS "OpenBLAS configured from binaries at ${openblas_binaries_SOURCE_DIR}")
            
            # Install the DLL so it ends up in the wheel/install directory
            install(FILES "${openblas_binaries_SOURCE_DIR}/bin/libopenblas.dll" DESTINATION bin)
            install(FILES "${openblas_binaries_SOURCE_DIR}/bin/libopenblas.dll" DESTINATION pycauset)
        endif()
        
    else()
        # On Linux/Mac, try to build from source (requires Fortran)
        message(STATUS "Downloading OpenBLAS source...")
        FetchContent_Declare(
            OpenBLAS
            URL https://github.com/OpenMathLib/OpenBLAS/archive/v0.3.26.zip
        )
        FetchContent_MakeAvailable(OpenBLAS)
    endif()
endif()

# 3.9 LAPACKE Header Fix (Required on some Linux distros)
if(UNIX AND NOT APPLE)
    find_package(PkgConfig QUIET)
    pkg_check_modules(PC_LAPACKE lapacke)
    if(PC_LAPACKE_FOUND)
        message(STATUS "Found LAPACKE via PkgConfig: ${PC_LAPACKE_INCLUDE_DIRS}")
        include_directories(SYSTEM ${PC_LAPACKE_INCLUDE_DIRS})
    endif()
endif()

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
    src/core/DebugTrace.cpp
    src/core/IOAccelerator.cpp
    src/core/PersistentObject.cpp
    src/core/StorageUtils.cpp
    src/core/SystemUtils.cpp
    src/core/ParallelUtils.cpp
    src/core/PromotionResolver.cpp
    src/matrix/MatrixBase.cpp
    src/matrix/MatrixOps.cpp
    src/matrix/DenseBitMatrix.cpp
    src/matrix/TriangularMatrix.cpp
    src/matrix/TriangularBitMatrix.cpp
    src/core/ObjectFactory.cpp
    src/vector/VectorBase.cpp
    src/vector/DenseVector.cpp
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
target_link_libraries(pycauset_core PUBLIC Eigen3::Eigen OpenBLAS::OpenBLAS)
if(PC_LAPACKE_FOUND)
    target_link_libraries(pycauset_core PUBLIC ${PC_LAPACKE_LINK_LIBRARIES})
endif()
set_property(TARGET pycauset_core PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(pycauset_core PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(WIN32 AND DEFINED PYCAUSET_OPENBLAS_DLL AND EXISTS "${PYCAUSET_OPENBLAS_DLL}")
    add_custom_command(
        TARGET pycauset_core
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${PYCAUSET_OPENBLAS_DLL}"
                $<TARGET_FILE_DIR:pycauset_core>
        VERBATIM
    )
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(pycauset_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(pycauset_core PRIVATE /bigobj)
endif()

# --- Extension Module ---
# Note: The module name must match what is defined in PYBIND11_MODULE in bindings.cpp (which is "_pycauset")
# Touch 2
python_add_library(_pycauset MODULE
    src/bindings.cpp
    src/bindings/bind_core.cpp
    src/bindings/binding_warnings.cpp
    src/bindings/bind_matrix.cpp
    src/bindings/bind_expression.cpp
    src/bindings/bind_vector.cpp
    src/bindings/bind_causet.cpp
)

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
option(ENABLE_CUDA "Enable CUDA acceleration" OFF)

if(ENABLE_CUDA)
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
        
        # Suppress specific NVCC warnings
        # 611: overloaded virtual function is only partially overridden
        # 177: variable was declared but never referenced
        # Note: We pass these flags individually without SHELL: prefix because CMake 3.24+ handles them correctly
        # and the SHELL: prefix combined with generator expressions can sometimes cause quoting issues with NVCC on Windows.
        target_compile_options(pycauset_cuda PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe>
            $<$<COMPILE_LANGUAGE:CUDA>:--diag_suppress=611>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe>
            $<$<COMPILE_LANGUAGE:CUDA>:--diag_suppress=177>
        )
        
        install(TARGETS pycauset_cuda DESTINATION pycauset)
    else()
        message(STATUS "CUDA not found. Skipping pycauset_cuda accelerator.")
    endif()
else()
    message(STATUS "CUDA disabled by configuration.")
endif()

# --- Installation ---
# This installs the compiled extension into the python package directory
install(
    TARGETS pycauset_core
    RUNTIME DESTINATION pycauset
    LIBRARY DESTINATION pycauset
    ARCHIVE DESTINATION pycauset
)
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

    # Symmetric Matrix Tests
    add_executable(test_symmetric_matrix tests/test_symmetric_matrix.cpp)
    target_include_directories(test_symmetric_matrix PRIVATE include)
    target_link_libraries(test_symmetric_matrix PRIVATE GTest::gtest_main pycauset_core)
    add_test(NAME SymmetricMatrixTests COMMAND test_symmetric_matrix)

    # Lazy Evaluation Tests
    add_executable(test_lazy_evaluation tests/test_lazy_evaluation.cpp)
    target_include_directories(test_lazy_evaluation PRIVATE include)
    target_link_libraries(test_lazy_evaluation PRIVATE GTest::gtest_main pycauset_core)
    add_test(NAME LazyEvaluationTests COMMAND test_lazy_evaluation)

    # Comprehensive Lazy Tests
    add_executable(test_lazy_comprehensive tests/test_lazy_comprehensive.cpp)
    target_include_directories(test_lazy_comprehensive PRIVATE include)
    target_link_libraries(test_lazy_comprehensive PRIVATE GTest::gtest_main pycauset_core)
    add_test(NAME LazyComprehensiveTests COMMAND test_lazy_comprehensive)

    # Other tests disabled for now as source files are missing in this context
    # add_executable(causal_tests tests/test_main.cpp tests/test_parallel.cpp tests/test_gpu.cpp)
    # ...

endif()

# --- Benchmarks ---
# add_executable(benchmark_native benchmarks/benchmark_native.cpp)
# target_include_directories(benchmark_native PRIVATE include)
# target_link_libraries(benchmark_native PRIVATE pycauset_core)
