#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "magma" for configuration "Release"
set_property(TARGET magma APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(magma PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmagma.so"
  IMPORTED_SONAME_RELEASE "libmagma.so"
  )

list(APPEND _cmake_import_check_targets magma )
list(APPEND _cmake_import_check_files_for_magma "${_IMPORT_PREFIX}/lib/libmagma.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
