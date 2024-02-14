#!/bin/bash

ml rocm/5.7.0
ml rocmcc/5.7.0-magic
ml cmake/3.24.2
ml python/3.9.12

cc=hipcc
cpp=hipcc

BOBA_DIR=$(pwd)
BUILD_DIR=${BOBA_DIR}/build_hip

mkdir -p $BUILD_DIR
cd $BUILD_DIR
cmake \
 -DCMAKE_INSTALL_PREFIX=${BOBA_DIR}/install/example_hip \
 -DMAGMA_DIR=${BOBA_DIR}/../install/magma_hip/lib/cmake/magma \
 -DCMAKE_CXX_COMPILER=$cpp \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_VERBOSE_MAKEFILE=ON \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
 ..
