cmake_minimum_required(VERSION 3.20)
project(pycusp LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(nanobind CONFIG REQUIRED)

set(CCCL_PATH "$ENV{CCCL_PATH}" CACHE PATH "Path to CCCL source tree")
if(NOT CCCL_PATH)
  message(FATAL_ERROR
    "CCCL_PATH is not set. Set the CCCL_PATH environment variable "
    "(or -DCCCL_PATH=...) to the root of a CCCL checkout, e.g. ~/cccl.")
endif()
foreach(sub cub libcudacxx/include thrust)
  if(NOT EXISTS "${CCCL_PATH}/${sub}")
    message(FATAL_ERROR "Missing ${CCCL_PATH}/${sub}; check CCCL_PATH.")
  endif()
endforeach()

set(CUSP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
if(NOT EXISTS "${CUSP_ROOT}/cusp/csr_matrix.h")
  message(FATAL_ERROR "Cannot locate CUSP headers at ${CUSP_ROOT}/cusp/")
endif()

nanobind_add_module(_core
  NB_STATIC STABLE_ABI
  src/_core.cpp)

target_include_directories(_core PRIVATE
  "${CCCL_PATH}/cub"
  "${CCCL_PATH}/libcudacxx/include"
  "${CCCL_PATH}/thrust"
  "${CUSP_ROOT}")

target_compile_definitions(_core PRIVATE
  THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
  CUSP_HOST_BLAS_SYSTEM=CUSP_HOST_BLAS_GENERIC
  CUSP_DEVICE_BLAS_SYSTEM=CUSP_DEVICE_BLAS_GENERIC)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(_core PRIVATE -Wno-unused-local-typedef)
endif()

install(TARGETS _core LIBRARY DESTINATION pycusp)
