# Define the minimum CMake version required
cmake_minimum_required(VERSION 3.31)

# Define the project name
project(mnist_from_scratch)

set(CMAKE_CXX_STANDARD 17)
# Include root directory
include_directories(./src/)
# Include sub modules
# Data Loader
include_directories(./src/data_loader/)
include_directories(./src/data_loader/base)
# Linear Algebra
include_directories(./src/lin_alg)
# Neural Network packages
include_directories(./src/nn)
include_directories(./src/nn/activations/)
include_directories(./src/nn/activations/base)
include_directories(./src/nn/layers/)
include_directories(./src/nn/layers/base)
include_directories(./src/nn/loss/)
include_directories(./src/nn/loss/base)


# Model implementations
include_directories(./src/model/)
include_directories(./src/model/base)

add_executable(RunMnistFromScratch
    ./src/main.cpp
        src/lin_alg/matrix.cpp
)