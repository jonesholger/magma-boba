
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was magmaConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

if(OFF)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
elseif(ON)
    if(NOT /opt/rocm-6.1.1)
    # First try finding paths given by the user
        find_path(ROCM_PATH
            hip
            PATHS
            $ENV{ROCM_DIR}
            $ENV{ROCM_PATH}
            $ENV{HIP_PATH}
            ${HIP_PATH}/..
            ${HIP_ROOT_DIR}/../
            ${ROCM_ROOT_DIR}
            /opt/rocm
            NO_DEFAULT_PATH
            NO_CMAKE_ENVIRONMENT_PATH
            NO_CMAKE_PATH
            NO_SYSTEM_ENVIRONMENT_PATH
            NO_CMAKE_SYSTEM_PATH)

    # If that fails, use CMake default paths
        if(NOT ROCM_PATH)
            find_path(ROCM_PATH hip)
        endif()
    endif()

    # Update CMAKE_PREFIX_PATH to make sure all the configs that hip depends on are
    # found.
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${ROCM_PATH};${ROCM_ROOT_DIR}/lib/cmake")

    find_package(hip REQUIRED CONFIG PATHS  ${HIP_PATH} ${ROCM_PATH} ${ROCM_ROOT_DIR}/lib/cmake/hip)
    find_package(hipblas REQUIRED CONFIG PATHS  ${HIP_PATH} ${ROCM_PATH} ${ROCM_ROOT_DIR}/lib/cmake/hip)
    find_package(hipsparse REQUIRED CONFIG PATHS  ${HIP_PATH} ${ROCM_PATH} ${ROCM_ROOT_DIR}/lib/cmake/hip)
    message(STATUS "ROCM path:        ${ROCM_PATH}")
    message(STATUS "HIP version:      ${hip_VERSION}")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/magmaTargets.cmake")
check_required_components( magma )
set_and_check(MAGMA_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../include)
set_target_properties(magma PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${MAGMA_INCLUDE_DIR}
)

