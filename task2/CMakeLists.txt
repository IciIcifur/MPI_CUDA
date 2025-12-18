cmake_minimum_required(VERSION 3.18)
project(MPI_CUDA_Task2 LANGUAGES C CXX)

option(BUILD_CPU "Build CPU variant" ON)
option(BUILD_CUDA "Build CUDA variant" ON)

find_package(OpenMP QUIET)
if (OpenMP_C_FOUND)
    message(STATUS "Found OpenMP")
    set(HAVE_OPENMP TRUE)
endif()

find_package(CUDA QUIET)
if (CUDA_FOUND)
    message(STATUS "Found CUDA")
endif()

set(SRC_DIR "${CMAKE_SOURCE_DIR}/../src")
message(STATUS "SRC_DIR = ${SRC_DIR}")

if (BUILD_CPU)
    add_executable(n_body_cpu ${SRC_DIR}/n_body_cpu.c)
    if (HAVE_OPENMP)
        target_compile_options(n_body_cpu PRIVATE $<IF:$<C_COMPILER_ID:MSVC>,/openmp,-fopenmp>)
    endif()
    target_include_directories(n_body_cpu PRIVATE ${SRC_DIR})
endif()

if (BUILD_CUDA)
    enable_language(CUDA)
    add_executable(n_body_cuda ${SRC_DIR}/n_body_cuda.cu)
    target_include_directories(n_body_cuda PRIVATE ${SRC_DIR})
endif()