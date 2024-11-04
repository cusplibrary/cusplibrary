# notes:
# to set the install at configuration time:
#   cmake -S $src_dir -B $build_dir -DCMAKE_INSTALL_PREFIX=$install_dir
#   cmake -S $src_dir -B $build_dir --install-prefix=$install_dir # Since CMake 3.21
# to set the install at install time
#   cmake --install $build_dir --prefix=$install_dir 
cmake_minimum_required(VERSION 3.29)

project(
    cusp
    VERSION 0.6.0
    DESCRIPTION "CUSP : A C++ Templated Sparse Matrix Library"
    HOMEPAGE_URL "https://github.com/cusplibrary/cusplibrary"
    LANGUAGES CXX CUDA
)

# Library definitions
# ------------

add_library(cusp INTERFACE)
target_include_directories(
    cusp INTERFACE
    "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cusp>"
    "\$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

# Version Info
# ------------

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "cuspConfigVersion.cmake"
    COMPATIBILITY SameMinorVersion
    ARCH_INDEPENDENT
)

# Install directions
# --------------------

include(GNUInstallDirs)
set(package cusp)

# copy the cusp header directory
install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/cusp
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# make an export target
install(
    TARGETS cusp
    EXPORT cuspTargets
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# install cuspConfig.cmake
# install version file cuspConfigVersion.cmake
install(
    FILES cmake/cuspConfig.cmake
          "${PROJECT_BINARY_DIR}/cuspConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/cusp/"
)

# install cuspTargets.cmake
install(
    EXPORT cuspTargets
    FILE cuspTargets.cmake
    NAMESPACE cusp::
    DESTINATION "${CMAKE_INSTALL_DATADIR}/cusp"
)

# Cpack
# -----
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")
include(CPack)

#option(BUILD_EXAMPLES "Build examples." "ON")
#if(BUILD_EXAMPLES)
#    add_subdirectory(examples)
#endif()
#
#include(cmake/dev-mode.cmake)
