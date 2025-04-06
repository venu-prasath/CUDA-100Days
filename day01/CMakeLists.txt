cmake_minimum_required(VERSION 3.8)
project(vector_add CUDA CXX)

# Enable CUDA
enable_language(CUDA)

# Add the executable
add_executable(vector_add vector_add_.cu)

# Set CUDA specific flags
set_target_properties(vector_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Set C++ standard
set_target_properties(vector_add PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
) 