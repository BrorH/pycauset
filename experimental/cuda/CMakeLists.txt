# Experimental CUDA accelerator (on hold). Included from the top-level CMakeLists.txt
# only with -DENABLE_CUDA=ON. Not built in CI. See README.md.
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        # Default to the build machine's GPU ("native"), but allow distribution
        # builds (CI) to override with a list of target compute capabilities
        # (e.g. "61;70;75;80;86;89") so the wheel runs on a range of GPUs and
        # does not require a GPU on the builder.
        set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA compute capabilities to target")
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        
        message(STATUS "CUDA found. Building pycauset_cuda accelerator.")
        
        add_library(pycauset_cuda SHARED
            ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDevice.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaSolver.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/src/MatmulDriver.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/src/CholeskyDriver.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/src/ArnoldiDriver.cu
        )
        
        target_include_directories(pycauset_cuda PRIVATE ${PROJECT_SOURCE_DIR}/include)
        
        # Link against CUDA libraries
        target_link_libraries(pycauset_cuda PRIVATE 
            CUDA::cudart 
            CUDA::cublas 
            CUDA::cusolver
            pycauset_core
            Eigen3::Eigen
        )
        
        # Set properties
        set_target_properties(pycauset_cuda PROPERTIES 
            WINDOWS_EXPORT_ALL_SYMBOLS ON
            CUDA_SEPARABLE_COMPILATION ON
        )

        # Eigen's C++11 math detection mis-fires under nvcc's host compilation
        # (EIGEN_USING_STD(arg) falls back to `using ::arg`, which does not exist),
        # so force the std::arg path.
        target_compile_definitions(pycauset_cuda PRIVATE EIGEN_HAS_CXX11_MATH=1)
        
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
        
        # --- Python Bindings for CUDA (Internal) ---
        # Only meaningful if CUDA is enabled. This module provides discovery functions.
        python_add_library(_pycauset_cuda MODULE ${CMAKE_CURRENT_SOURCE_DIR}/bindings/bind_cuda.cpp)
        target_include_directories(_pycauset_cuda PRIVATE ${PROJECT_SOURCE_DIR}/include)
        target_link_libraries(_pycauset_cuda PRIVATE pycauset_cuda pybind11::module pycauset_core CUDA::cudart)
        set_target_properties(_pycauset_cuda PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
        install(TARGETS _pycauset_cuda DESTINATION pycauset)

        install(TARGETS pycauset_cuda DESTINATION pycauset)
    else()
        message(STATUS "CUDA not found. Skipping pycauset_cuda accelerator.")
    endif()
