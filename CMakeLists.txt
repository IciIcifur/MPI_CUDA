cmake_minimum_required(VERSION 3.18)
project(MPI_CUDA C)

set(CMAKE_C_STANDARD 11)

find_package(OpenMP REQUIRED)

# Mandelbrot (OpenMP)
add_executable(mandelbrot src/mandelbrot.c)
target_link_libraries(mandelbrot PUBLIC OpenMP::OpenMP_C)

# N-body CPU
add_executable(nbody_cpu src/n_body_cpu.c)
target_link_libraries(nbody_cpu PUBLIC OpenMP::OpenMP_C)

# N-body CUDA
# enable_language(CUDA)
# add_executable(nbody_cuda src/n_body_cuda.cu)
# set_target_properties(nbody_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)