cmake_minimum_required(VERSION 3.21)
project(CUDAProgramming CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(CUDAProgramming main.cu)

set_target_properties(CUDAProgramming PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
