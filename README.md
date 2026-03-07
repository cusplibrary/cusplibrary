## CUSP : A C++ Templated Sparse Matrix Library

| CI | Coverage |
| -- | -------- |
| [![CI](https://github.com/cusplibrary/cusplibrary/actions/workflows/ci.yml/badge.svg)](https://github.com/cusplibrary/cusplibrary/actions/workflows/ci.yml) | [![Coverage](https://codecov.io/gh/cusplibrary/cusplibrary/branch/main/graph/badge.svg)](https://codecov.io/gh/cusplibrary/cusplibrary) |

For more information, see the project documentation at [CUSP Website](http://cusplibrary.github.io).

### A Simple Example

```C++
#include <cuda.h>
#include <thrust/version.h>

#include <cusp/version.h>
#include <cusp/hyb_matrix.h>
#include <cusp/io/matrix_market.h>
#include <cusp/krylov/cg.h>

#include <iostream>

int main(void)
{
    int cuda_major =  CUDA_VERSION / 1000;
    int cuda_minor = (CUDA_VERSION % 1000) / 10;
    int thrust_major = THRUST_MAJOR_VERSION;
    int thrust_minor = THRUST_MINOR_VERSION;
    int cusp_major = CUSP_MAJOR_VERSION;
    int cusp_minor = CUSP_MINOR_VERSION;
    std::cout << "CUDA   v" << cuda_major   << "." << cuda_minor   << std::endl;
    std::cout << "Thrust v" << thrust_major << "." << thrust_minor << std::endl;
    std::cout << "Cusp   v" << cusp_major   << "." << cusp_minor   << std::endl;

    // create an empty sparse matrix structure (HYB format)
    cusp::hyb_matrix<int, float, cusp::device_memory> A;

    // load a matrix stored in Matrix-Market format
    cusp::io::read_matrix_market_file(A, "./testing/data/laplacian/5pt_10x10.mtx");

    // allocate storage for solution (x) and right hand side (b)
    cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0);
    cusp::array1d<float, cusp::device_memory> b(A.num_rows, 1);

    // solve the linear system A * x = b with the conjugate gradient method
    cusp::krylov::cg(A, x, b);

    return 0;
}
```

 CUSP is a header-only library.  To compile this example clone both CUSP and [Nvidia/cccl](https://github.com/NVIDIA/cccl):
```shell
git@github.com:cusplibrary/cusplibrary.git
cd cusplibrary
git clone git@github.com:NVIDIA/cccl.git
nvcc -Icccl/thrust -Icccl/libcudacxx/include -Icccl/cub -I. example.cu -o example
```

### Status

CUSP is up-to-date with CCCL v2.8.5.  It has been tested with CUDA 13.

### To build:

First point to CCCL (and to THRUST if using a `gcc` build).  For example with CCCL in your home directory:
```shell
export CCCL_PATH=$HOME/cccl
export THRUST_PATH=$HOME/cccl/thrust
```

Next, you must have a recent version of Scons for the build:
```shell
pip install scons
```

To build/run with gcc, use an `omp` backend with CUSP blas:
```shell
scons compiler=gcc backend=omp deviceblas=cusp mode=release -j 16
```

To build/run with nvcc, use:
```shell
scons arch=sm_90 -j 16
```
(see https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/ for a full list)

A single test can be run with `scons single_test=poisson`, for example.

Examples can be found in `examples` and performance results can be found in `performance` with similar scons instructions.

### Contributors

CUSP is developed as an open-source project with [NVIDIA Research](https://research.nvidia.com).
[Nathan Bell](https:github.com/wnbell) was the original creator.
It is currently developed by 
[Steven Dalton](https://github.com/sdalton1) and
[Luke Olson](https://github.com/lukeolson)

CUSP is available under the Apache v2.0 open source [LICENSE](./LICENSE)

### Citing

```shell
@MISC{Cusp,
  author = "Steven Dalton and Nathan Bell and Luke Olson and Michael Garland",
  title = "Cusp: Generic Parallel Algorithms for Sparse Matrix and Graph Computations",
  year = "2026",
  url = "https://github.com/cusplibrary/cusplibrary",
  note = "Version 0.6.0"
}
```
