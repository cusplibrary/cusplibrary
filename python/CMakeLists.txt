cmake_minimum_required(VERSION 3.24)
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

# ---------------------------------------------------------------------------
# Optional CUDA support
# ---------------------------------------------------------------------------
set(PYCUSP_CUDA "AUTO" CACHE STRING
    "Build pycusp with CUDA support (ON/OFF/AUTO)")

set(_pycusp_enable_cuda OFF)
if(PYCUSP_CUDA STREQUAL "AUTO")
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    set(_pycusp_enable_cuda ON)
  endif()
elseif(PYCUSP_CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(_pycusp_enable_cuda ON)
endif()

set(PYCUSP_SOURCES src/_core.cpp)
if(_pycusp_enable_cuda)
  enable_language(CUDA)
  list(APPEND PYCUSP_SOURCES src/_cuda.cu)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 90)
  endif()
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  message(STATUS "pycusp: CUDA enabled (archs: ${CMAKE_CUDA_ARCHITECTURES})")
else()
  message(STATUS "pycusp: CUDA disabled (host-only build)")
endif()

nanobind_add_module(_core
  NB_STATIC STABLE_ABI
  ${PYCUSP_SOURCES})

target_include_directories(_core PRIVATE
  "${CCCL_PATH}/cub"
  "${CCCL_PATH}/libcudacxx/include"
  "${CCCL_PATH}/thrust"
  "${CUSP_ROOT}")

# CPU translation unit: sequential host dispatch only
set_source_files_properties(src/_core.cpp PROPERTIES
  COMPILE_DEFINITIONS "THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP")

target_compile_definitions(_core PRIVATE
  CUSP_HOST_BLAS_SYSTEM=CUSP_HOST_BLAS_GENERIC
  CUSP_DEVICE_BLAS_SYSTEM=CUSP_DEVICE_BLAS_GENERIC)

if(_pycusp_enable_cuda)
  target_compile_definitions(_core PRIVATE PYCUSP_HAVE_CUDA=1)
  set_source_files_properties(src/_cuda.cu PROPERTIES
    COMPILE_DEFINITIONS "THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA")
  target_link_libraries(_core PRIVATE CUDA::cudart)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-local-typedef>)
endif()

install(TARGETS _core LIBRARY DESTINATION pycusp)
