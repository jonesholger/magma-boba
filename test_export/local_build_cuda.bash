#!/bin/bash
ml gcc/8.3.1
ml cuda/11.8.0
ml cmake/3.23.1
ml python/3.8.2

cpp=g++

BOBA_DIR=$(pwd)
BUILD_DIR=${BOBA_DIR}/build_cuda

mkdir -p $BUILD_DIR
cd $BUILD_DIR
cmake \
 -DCMAKE_INSTALL_PREFIX=${BOBA_DIR}/install/example_cuda \
 -DMAGMA_DIR=${BOBA_DIR}/../install/magma_cuda/lib/cmake/magma \
 -DCMAKE_CXX_COMPILER=$cpp \
 -DENABLE_CUDA=ON \
 -DCMAKE_CUDA_ARCHITECTURES="70" \
 -DCMAKE_CUDA_COMPILER="nvcc" \
 -DCMAKE_CXX_STANDARD="17" \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_VERBOSE_MAKEFILE=ON \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
 ..
