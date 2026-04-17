\page quickstart Quick Start Guide
# Summary Guide for new Cusp developers

## Introduction

This page describes how to develop CUDA applications with CUSP, a C++ template library for sparse matrix computations.

## Prerequisites

CUSP requires CUDA. Check the CUDA installation by using nvcc on the command line as follows:

```shell
nvcc --version
```

CUSP is a template library and requires no "build".  Clone or download the latest version of [CUSP](https://github.com/cusplibrary/cusplibrary/releases).

## Example

The following example will check that the prerequisites are satisfied.  Save the following source code to a file named `version.cu`.

```cpp
#include <cuda.h>
#include <thrust/version.h>
#include <cusp/version.h>

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

    return 0;
}
```

Set the CUSP and THRUST (CCCL) paths:
```shell
export CCCL_PATH=$HOME/somepath
export CUSP_PATH=$HOME/somepath
```

Now compile `version.cu` with `nvcc`:

```shell
nvcc version.cu -o version -I$CUSP_PATH
nvcc version.cu -o version
```

## Other Examples

Additional CUSP examples are available at https://github.com/cusplibrary/cusplibrary/tree/main/examples can be compiled using the same steps as above.  For instance, the conjugate gradient solver example is compiled and run as follows:

```shell
cd examples/Solvers/
nvcc -O2 cg.cu -o cg
```

Then
```shell
./cg
Solver will continue until residual norm 0.01 or reaching 100 iterations
Iteration Number  | Residual Norm
            0       1.000000e+01
            1       1.414214e+01
            2       1.093707e+01
            3       8.949319e+00
            4       6.190055e+00
            5       3.835189e+00
            6       1.745481e+00
            7       5.963546e-01
            8       2.371134e-01
            9       1.152524e-01
           10       3.134467e-02
           11       1.144415e-02
           12       1.824176e-03
Successfully converged after 12 iterations.
```

## Sparse Matrices

Cusp natively supports several sparse matrix formats:
  * [Coordinate (COO)](classcusp_1_1coo__matrix.html)
  * [Compressed Sparse Row (CSR)](classcusp_1_1csr__matrix.html)
  * [Diagonal (DIA)](classcusp_1_1dia__matrix.html)
  * [ELL (ELL)](classcusp_1_1ell__matrix.html)
  * [Hybrid (HYB)](classcusp_1_1hyb__matrix.html)
  * [Permutation](classcusp_1_1permutation__matrix.html)

When manipulating matrices it is important to consider the advantages and disadvantages of each format.  In general, the DIA and ELL formats are the most efficient for computing sparse matrix-vector products, and therefore are the fastest formats for solving sparse linear systems with iterative methods (e.g. conjugate gradient).  The COO and CSR formats are more flexible than DIA and ELL and easier manipulate.  The HYB format is a hybrid combination of the ELL (fast) and COO (flexible) formats and is a good default choice.  Refer to the matrix format [examples](examples.html) for additional information.

## Format Conversions

CUSP facilitates the transfer of data between the host and device and the conversion between sparse matrix formats.  For example,

```cpp
#include <cusp/csr_matrix.h>
#include <cusp/hyb_matrix.h>

int main()
{
  // Allocate storage for a 5 by 8 sparse matrix in CSR format with 12
  // nonzero entries on the host
  cusp::csr_matrix<int,float,cusp::host_memory> csr_host(5,8,12);

  // Transfer the matrix to the device
  cusp::csr_matrix<int,float,cusp::device_memory> csr_device(csr_host);

  // Convert the matrix to HYB format on the device
  cusp::hyb_matrix<int,float,cusp::device_memory> hyb_device(csr_device);
}
```

## Iterative Solvers

Cusp provides a variety of [iterative methods](https://en.wikipedia.org/wiki/Iterative_method) for solving sparse linear systems.  Established [Krylov subspace](https://en.wikipedia.org/wiki/Krylov_subspace)
methods are available:
  * [Conjugate-Gradient(CG)](group__krylov__methods.html#ga6aa97799b77e1de21fc88be236c6e4a8)
  * [Conjugate-Residual(CR)](group__krylov__methods.html#gae73aeb8fd04ee86a128240ac62c20e33)
  * [Biconjugate Gradient (BiCG)](group__krylov__methods.html#gad82e975fa15cb096d13507163325c2b5)
  * [Biconjugate Gradient Stabilized (BiCGstab)](group__krylov__methods.html#ga23cfa8325966505d6580151f91525887)
  * [Generalized Minimum Residual (GMRES)](group__krylov__methods.html#ga691b2d4d03fd7b23e674f9f046691b46)
  * [Multi-mass Conjugate-Gradient (CG-M)](group__krylov__methods.html#gae25c1e3e77e92709bfa9f3726328e421)
  * [Multi-mass Biconjugate Gradient stabilized (BiCGstab-M)](group__krylov__methods.html#gae9649279f0fb30cbc6a48c9f912a5f87)

More detailed examples are available [here](examples.html).

## Preconditioners

CUSP provides the following [preconditioners](https://en.wikipedia.org/wiki/Preconditioner):

  * [Algebraic Multigrid (AMG) based on Smoothed Aggregation](classcusp_1_1precond_1_1aggregation_1_1smoothed__aggregation.html)
  * [Approximate Inverse(AINV)](classcusp_1_1precond_1_1bridson__ainv.html)
  * [Diagonal](classcusp_1_1precond_1_1diagonal.html)

## User-Defined Linear Operators

It may be useful to solve a linear system `A x = b` without converting the matrix `A` into a CUSP matrix format.  CUSP supports user-defined [linear operators](classcusp_1_1linear__operator.html) that take a vector `x` as input and compute the matrix-vector product `y = A * x` as output.  These operators can be used with the iterative solvers in CUSP.

## Additional Resources

  * Comprehensive [Documentation](modules.html) of Cusp's API
  * A list of [Frequently Asked Questions](https://code.google.com/p/cusp-library/wiki/FrequentlyAskedQuestions)
  * Collection of [example](examples.html) programs

<!-- include Algorithms examples  -->
\example blas.cu
\example multiply.cu
\example maximal_independent_set.cu
\example transpose.cu
<!-- include Gallery examples  -->
\example diffusion.cu
\example poisson.cu
<!-- include InputOutput examples  -->
\example matrix_market.cu
<!-- include LinearOperator examples  -->
\example stencil.cu
<!-- include MatrixAssembly examples  -->
\example unordered_triplets.cu
<!-- include MatrixFormats examples  -->
\example coo.cu
\example csr.cu
\example dia.cu
\example ell.cu
\example hyb.cu
<!-- include Monitor examples  -->
\example monitor.cu
\example verbose_monitor.cu
<!-- include Preconditioner examples  -->
\example ainv.cu
\example diagonal.cu
\example smoothed_aggregation.cu
\example custom_amg.cu
<!-- include Solver examples  -->
\example bicgstab.cu
\example cg.cu
\example cg_m.cu
\example cr.cu
\example gmres.cu
<!-- include View examples  -->
\example array1d.cu
\example array2d_raw.cu
\example cg_raw.cu
\example csr_view.cu
\example csr_raw.cu
