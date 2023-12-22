## CUSP : A C++ Templated Sparse Matrix Library

| Linux | Windows | Coverage |
| ----- | ------- | -------- |
| [![Linux](https://travis-ci.org/sdalton1/cusplibrary.png)](https://travis-ci.org/sdalton1/cusplibrary) | [![Windows](https://ci.appveyor.com/api/projects/status/36pf1oqwkfq6xekn?svg=true)](https://ci.appveyor.com/project/StevenDalton/cusplibrary) | [![Coverage](https://coveralls.io/repos/sdalton1/cusplibrary/badge.svg?branch=master)](https://coveralls.io/r/sdalton1/cusplibrary?branch=master) |

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

### Stable Releases

CUSP releases are labeled using version identifiers having three fields:

| Date | Version | Date | Version |
| ---- | ------- | ---- | ------- |
|            |                                                                              | 03/13/2015 | [CUSP v0.5.0](https://github.com/cusplibrary/cusplibrary/archive/v0.5.0.zip) |
|            |                                                                              | 08/30/2013 | [CUSP v0.4.0](https://github.com/cusplibrary/cusplibrary/archive/v0.4.0.zip) |
|            |                                                                              | 03/08/2012 | [CUSP v0.3.1](https://github.com/cusplibrary/cusplibrary/archive/v0.3.1.zip) |
|            |                                                                              | 02/04/2012 | [CUSP v0.3.0](https://github.com/cusplibrary/cusplibrary/archive/v0.3.0.zip) |
|            |                                                                              | 05/30/2011 | [CUSP v0.2.0](https://github.com/cusplibrary/cusplibrary/archive/v0.2.0.zip) |
| 04/28/2015 | [CUSP v0.5.1](https://github.com/cusplibrary/cusplibrary/archive/v0.5.1.zip) | 07/10/2010 | [CUSP v0.1.0](https://github.com/cusplibrary/cusplibrary/archive/v0.1.0.zip) |


### Contributors

CUSP is developed as an open-source project by [NVIDIA Research](http://research.nvidia.com).
[Nathan Bell](http:github.com/wnbell) was the original creator and
[Steven Dalton](http://github.com/sdalton1) is the current primary contributor.

CUSP is available under the Apache v2.0 open source [LICENSE](./LICENSE)

### Citing

```shell
@MISC{Cusp,
  author = "Steven Dalton and Nathan Bell and Luke Olson and Michael Garland",
  title = "Cusp: Generic Parallel Algorithms for Sparse Matrix and Graph Computations",
  year = "2014",
  url = "http://cusplibrary.github.io/", note = "Version 0.5.0"
}
```
