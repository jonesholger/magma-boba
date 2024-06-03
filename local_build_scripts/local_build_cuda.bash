#!/bin/bash
ml gcc/8.3.1
ml cuda/11.8.0
ml cmake/3.23.1
ml python/3.8.2

COMP_NVCC_VER=11.8.0
COMP_ARCH=sm_70
COMP_GCC_VER=8.3.1
BOBA_DIR=$(pwd)
BUILD_DIR=${BOBA_DIR}/build_cuda
RAJA_HOSTCONFIG=${BOBA_DIR}/tpl/RAJA/host-configs/lc-builds/blueos/nvcc_gcc_X.cmake

#git reset --hard d998fcbb94d9046ec98ae93757010a1472902d54
echo -e 'BACKEND=cuda\nGPU_TARGET=Volta\nFORT=false' > make.inc
make cleanall
make -j 32 generate

mkdir -p $BUILD_DIR
cd $BUILD_DIR
cmake \
 -DCMAKE_INSTALL_PREFIX=${BOBA_DIR}/install/magma_cuda \
 -DCMAKE_CXX_STANDARD="17" \
 -DMAGMA_ENABLE_CUDA=ON \
 -DBUILD_SHARED_LIBS=ON \
 -DGPU_TARGET=Volta \
 -DUSE_FORTRAN=OFF \
 -DMAGMA_TEST=OFF \
 -DMAGMA_SPARSE=OFF \
 -DQUIET=ON \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_VERBOSE_MAKEFILE=OFF \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
 -DLAPACK_LIBRARIES="/usr/lib64/liblapack.so;/usr/lib64/libessl.so" \
 ..

#make -j 32 magma
#make install -j 32

#echo "set PKG_CONFIG_PATH in env to pick up MAGMA"
#echo "PKG_CONFIG_PATH=./install/magma_cuda/lib/pkgconfig:$PKG_CONFIG_PATH"
#export PKG_CONFIG_PATH=${BOBA_DIR}/install/magma_cuda/lib/pkgconfig:$PKG_CONFIG_PATH