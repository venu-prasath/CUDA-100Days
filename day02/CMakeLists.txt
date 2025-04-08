cmake_minimum_required(VERSION 3.18)
project(MatrixAdditionCUDA LANGUAGES CXX CUDA)

# Specify the CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 75)

# Add the executable
add_executable(matrix_add matrix_add.cu)

# Set the C++ standard
set_target_properties(matrix_add PROPERTIES CXX_STANDARD 14 CUDA_STANDARD 14)