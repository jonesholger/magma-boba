#!/bin/bash

ml rocm/5.7.0
ml rocmcc/5.7.0-magic
ml cmake/3.24.2
ml python/3.9.12

cc=hipcc
cpp=hipcc
target=gfx90a

BOBA_DIR=$(pwd)
BUILD_DIR=${BOBA_DIR}/build_hip
#git reset --hard d998fcbb94d9046ec98ae93757010a1472902d54
echo -e 'BACKEND=hip\nGPU_TARGET=gfx90a\nFORT=false' > make.inc
make cleanall
make -j 32 generate

#-DCMAKE_EXE_LINKER_FLAGS="--offload-arch=$target" \
#-DCMAKE_HIP_ARCHITECTURES=$target \
#-DCMAKE_CXX_FLAGS="--offload-arch=$target" \
#-DGPU_TARGET=$target  \
# -DGPU_TARGETS=$target  \

mkdir -p $BUILD_DIR
cd $BUILD_DIR
cmake \
 -DCMAKE_INSTALL_PREFIX=${BOBA_DIR}/install/magma_hip \
 -DMAGMA_ENABLE_HIP=ON \
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
