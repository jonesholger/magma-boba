#!/bin/bash

ml rocm/6.0.3
ml rocmcc/6.0.3-magic
ml cmake/3.24.2
ml python/3.9.12

cc=hipcc
cpp=hipcc
target=gfx90a

BOBA_DIR=$(pwd)
BUILD_DIR=${BOBA_DIR}/build_hip
#echo -e 'BACKEND=hip\nGPU_TARGET=gfx90a\nFORT=false' > make.inc
#make cleanall
#make -j 32 generate

#MAGMA_ORIG=OFF is broken - doesn't setup memoryType in hipPointerAttribute_t
#MAGMA_ORIG below uses original HIP arch support while OFF is arch autodetect
#autodetect uses DetermineHIPCompiler to detect CMAKE_HIP_ARCHITECTURES
#warning Calls hipconfig as part of process and for rocm 6.1.1 location is wrong (temp)

mkdir -p $BUILD_DIR
cd $BUILD_DIR
cmake \
 -DCMAKE_INSTALL_PREFIX=${BOBA_DIR}/install/magma_hip \
 -DMAGMA_ENABLE_HIP=ON \
 -DMAGMA_ORIG=ON \
 -DCMAKE_CXX_COMPILER=$cpp \
 -DCMAKE_C_COMPILER=$cc \
 -DUSE_FORTRAN=OFF \
 -DMAGMA_TEST=OFF \
 -DMAGMA_SPARSE=OFF \
 -DQUIET=ON \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_VERBOSE_MAKEFILE=OFF \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
 ..
