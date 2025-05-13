# notes:
# to set the install at configuration time:
#   cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/path/to/install
#   cmake -S . -B build --install-prefix=/path/to/install
# to set the install at install time
#   cmake --install build --prefix=/path/to/install
# to set the arch
# cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=50
cmake_minimum_required(VERSION 3.29)

project(
    cusp
    VERSION 0.6.0
    DESCRIPTION "CUSP : A C++ Templated Sparse Matrix Library"
    HOMEPAGE_URL "https://github.com/cusplibrary/cusplibrary"
    LANGUAGES CXX CUDA
)

# Options
# -------
option(BUILD_MAIN "Build main example." "OFF")
option(BUILD_EXAMPLES "Build main example." "OFF")

# if building something, then install CCCL
if(BUILD_MAIN OR BUILD_EXAMPLES)
    include(cmake/CPM.cmake)
    CPMAddPackage(
        NAME CCCL
        GITHUB_REPOSITORY nvidia/cccl
        VERSION 2.6.1
    )
endif()

# Library definitions
# ------------
add_library(cusp INTERFACE)
add_library(cusp::cusp ALIAS cusp)
target_include_directories(
    cusp INTERFACE
    "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
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

# Builds
# ------
if(BUILD_MAIN)
    add_executable(example example.cu)
    file(COPY ${CMAKE_CURRENT_LIST_DIR}/testing/data/ DESTINATION ${CMAKE_BINARY_DIR}/testing//data/)
    target_link_libraries(example PUBLIC CCCL::CCCL)
    target_link_libraries(example PUBLIC cusp)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()


# Cpack
# -----
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")
include(CPack)
