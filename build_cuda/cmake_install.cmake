# Install script for directory: /g/g0/holger/workspace/magma

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/g/g0/holger/workspace/magma/install/magma_cuda")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmagma.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmagma.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmagma.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/g/g0/holger/workspace/magma/build_cuda/lib/libmagma.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmagma.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmagma.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmagma.so"
         OLD_RPATH "/usr/tce/packages/cuda/cuda-11.8.0/nvidia/lib64:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libmagma.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/g/g0/holger/workspace/magma/build_cuda/include/magma_config.h"
    "/g/g0/holger/workspace/magma/include/magma.h"
    "/g/g0/holger/workspace/magma/include/magma_auxiliary.h"
    "/g/g0/holger/workspace/magma/include/magma_batched.h"
    "/g/g0/holger/workspace/magma/include/magma_bulge.h"
    "/g/g0/holger/workspace/magma/include/magma_c.h"
    "/g/g0/holger/workspace/magma/include/magma_cbatched.h"
    "/g/g0/holger/workspace/magma/include/magma_cbulge.h"
    "/g/g0/holger/workspace/magma/include/magma_cbulgeinc.h"
    "/g/g0/holger/workspace/magma/include/magma_cgehrd_m.h"
    "/g/g0/holger/workspace/magma/include/magma_clapack.h"
    "/g/g0/holger/workspace/magma/include/magma_copy.h"
    "/g/g0/holger/workspace/magma/include/magma_copy_v1.h"
    "/g/g0/holger/workspace/magma/include/magma_cvbatched.h"
    "/g/g0/holger/workspace/magma/include/magma_d.h"
    "/g/g0/holger/workspace/magma/include/magma_dbatched.h"
    "/g/g0/holger/workspace/magma/include/magma_dbulge.h"
    "/g/g0/holger/workspace/magma/include/magma_dbulgeinc.h"
    "/g/g0/holger/workspace/magma/include/magma_dgehrd_m.h"
    "/g/g0/holger/workspace/magma/include/magma_dlapack.h"
    "/g/g0/holger/workspace/magma/include/magma_ds.h"
    "/g/g0/holger/workspace/magma/include/magma_dvbatched.h"
    "/g/g0/holger/workspace/magma/include/magma_hbatched.h"
    "/g/g0/holger/workspace/magma/include/magma_htc.h"
    "/g/g0/holger/workspace/magma/include/magma_lapack.h"
    "/g/g0/holger/workspace/magma/include/magma_mangling.h"
    "/g/g0/holger/workspace/magma/include/magma_mangling_cmake.h"
    "/g/g0/holger/workspace/magma/include/magma_operators.h"
    "/g/g0/holger/workspace/magma/include/magma_s.h"
    "/g/g0/holger/workspace/magma/include/magma_sbatched.h"
    "/g/g0/holger/workspace/magma/include/magma_sbulge.h"
    "/g/g0/holger/workspace/magma/include/magma_sbulgeinc.h"
    "/g/g0/holger/workspace/magma/include/magma_sgehrd_m.h"
    "/g/g0/holger/workspace/magma/include/magma_slapack.h"
    "/g/g0/holger/workspace/magma/include/magma_svbatched.h"
    "/g/g0/holger/workspace/magma/include/magma_types.h"
    "/g/g0/holger/workspace/magma/include/magma_v2.h"
    "/g/g0/holger/workspace/magma/include/magma_vbatched.h"
    "/g/g0/holger/workspace/magma/include/magma_z.h"
    "/g/g0/holger/workspace/magma/include/magma_zbatched.h"
    "/g/g0/holger/workspace/magma/include/magma_zbulge.h"
    "/g/g0/holger/workspace/magma/include/magma_zbulgeinc.h"
    "/g/g0/holger/workspace/magma/include/magma_zc.h"
    "/g/g0/holger/workspace/magma/include/magma_zgehrd_m.h"
    "/g/g0/holger/workspace/magma/include/magma_zlapack.h"
    "/g/g0/holger/workspace/magma/include/magma_zvbatched.h"
    "/g/g0/holger/workspace/magma/include/magmablas.h"
    "/g/g0/holger/workspace/magma/include/magmablas_c.h"
    "/g/g0/holger/workspace/magma/include/magmablas_c_v1.h"
    "/g/g0/holger/workspace/magma/include/magmablas_c_v1_map.h"
    "/g/g0/holger/workspace/magma/include/magmablas_d.h"
    "/g/g0/holger/workspace/magma/include/magmablas_d_v1.h"
    "/g/g0/holger/workspace/magma/include/magmablas_d_v1_map.h"
    "/g/g0/holger/workspace/magma/include/magmablas_ds.h"
    "/g/g0/holger/workspace/magma/include/magmablas_ds_v1.h"
    "/g/g0/holger/workspace/magma/include/magmablas_ds_v1_map.h"
    "/g/g0/holger/workspace/magma/include/magmablas_h.h"
    "/g/g0/holger/workspace/magma/include/magmablas_s.h"
    "/g/g0/holger/workspace/magma/include/magmablas_s_v1.h"
    "/g/g0/holger/workspace/magma/include/magmablas_s_v1_map.h"
    "/g/g0/holger/workspace/magma/include/magmablas_v1.h"
    "/g/g0/holger/workspace/magma/include/magmablas_v1_map.h"
    "/g/g0/holger/workspace/magma/include/magmablas_z.h"
    "/g/g0/holger/workspace/magma/include/magmablas_z_v1.h"
    "/g/g0/holger/workspace/magma/include/magmablas_z_v1_map.h"
    "/g/g0/holger/workspace/magma/include/magmablas_zc.h"
    "/g/g0/holger/workspace/magma/include/magmablas_zc_v1.h"
    "/g/g0/holger/workspace/magma/include/magmablas_zc_v1_map.h"
    "/g/g0/holger/workspace/magma/sparse/include/magmasparse.h"
    "/g/g0/holger/workspace/magma/sparse/include/magmasparse_c.h"
    "/g/g0/holger/workspace/magma/sparse/include/magmasparse_d.h"
    "/g/g0/holger/workspace/magma/sparse/include/magmasparse_ds.h"
    "/g/g0/holger/workspace/magma/sparse/include/magmasparse_mmio.h"
    "/g/g0/holger/workspace/magma/sparse/include/magmasparse_s.h"
    "/g/g0/holger/workspace/magma/sparse/include/magmasparse_types.h"
    "/g/g0/holger/workspace/magma/sparse/include/magmasparse_z.h"
    "/g/g0/holger/workspace/magma/sparse/include/magmasparse_zc.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/g/g0/holger/workspace/magma/build_cuda/lib/pkgconfig/magma.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/magma/magmaTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/magma/magmaTargets.cmake"
         "/g/g0/holger/workspace/magma/build_cuda/CMakeFiles/Export/lib/cmake/magma/magmaTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/magma/magmaTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/magma/magmaTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/magma" TYPE FILE FILES "/g/g0/holger/workspace/magma/build_cuda/CMakeFiles/Export/lib/cmake/magma/magmaTargets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/magma" TYPE FILE FILES "/g/g0/holger/workspace/magma/build_cuda/CMakeFiles/Export/lib/cmake/magma/magmaTargets-release.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/magma" TYPE FILE FILES
    "/g/g0/holger/workspace/magma/build_cuda/magmaConfig.cmake"
    "/g/g0/holger/workspace/magma/build_cuda/magmaConfigVersion.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/g/g0/holger/workspace/magma/build_cuda/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
