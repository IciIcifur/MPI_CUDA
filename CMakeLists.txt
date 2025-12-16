cmake_minimum_required(VERSION 3.18)
project(MPI_CUDA C)

set(CMAKE_C_STANDARD 11)

find_package(OpenMP REQUIRED)

# Mandelbrot
add_executable(mandelbrot src/mandelbrot.c)
target_link_libraries(mandelbrot PUBLIC OpenMP::OpenMP_C)

# N-body
add_executable(nbody_c src/n_body.c)
target_link_libraries(nbody_c PUBLIC OpenMP::OpenMP_C)

# CUDA
# enable_language(CUDA)
# add_executable(nbody_cuda src/n_body.cu)
# set_target_properties(nbody_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)