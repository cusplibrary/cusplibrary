/*
 *  Copyright 2008-2014 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file smoothed_aggregation.h
 *  \brief Algebraic multigrid preconditoner based on smoothed aggregation.
 *
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/detail/multilevel.h>

#include <cusp/eigen/spectral_radius.h>
#include <cusp/precond/aggregation/smoothed_aggregation_options.h>

#include <thrust/detail/use_default.h>

#include <vector> // TODO replace with host_vector

namespace cusp
{
namespace precond
{
namespace aggregation
{

/*! \addtogroup iterative_solvers Iterative Solvers
 *  \addtogroup preconditioners Preconditioners
 *  \ingroup iterative_solvers
 *  \{
 */

/**
 *  \brief Algebraic multigrid preconditoner based on smoothed aggregation
 *
 *  \tparam IndexType Type used for matrix values (e.g. \c int or \c size_t).
 *  \tparam ValueType Type used for matrix values (e.g. \c float or \c double).
 *  \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or \c cusp::device_memory)
 *
 *  \par Overview
 *  Given a matrix \c A to precondition, the smoothed aggregation preconditioner
 *  constructs a algebraic multigrid (AMG) operator.
 *
 *  Smoothed aggregation is expensive to use but is a very effective
 *  preconditioning technique to solve challenging linear systems.
 *  The default configuration uses a symmetric strength measure, MIS-based
 *  aggregation in device memory, sequential aggregation in host_memory,
 *  Jacobi smoothing is applied to the tentative prolongator, Jacobi relaxation
 *  on each level of hierarchy and LU to solve the coarse matrix in host
 *  memory.
 *
 *  \par Example
 *  The following code snippet demonstrates how to use a
 *  \p smoothed_aggregation preconditioner to solve a linear system.
 *
 *  \code
 *  #include <cusp/csr_matrix.h>
 *  #include <cusp/gallery/poisson.h>
 *  #include <cusp/io/matrix_market.h>
 *  #include <cusp/krylov/cg.h>
 *  #include <cusp/precond/aggregation/smoothed_aggregation.h>
 *
 *  #include <iostream>
 *
 *  int main(int argc, char *argv[])
 *  {
 *      typedef int                 IndexType;
 *      typedef double              ValueType;
 *      typedef cusp::device_memory MemorySpace;
 *
 *      // create an empty sparse matrix structure
 *      cusp::hyb_matrix<IndexType, ValueType, MemorySpace> A;
 *
 *      // construct 2d poisson matrix
 *      IndexType N = 256;
 *      cusp::gallery::poisson5pt(A, N, N);
 *
 *      std::cout << "Generated matrix (poisson5pt) "
 *                << "with shape ("  << A.num_rows << "," << A.num_cols << ") and "
 *                << A.num_entries << " entries" << "\n";
 *
 *      std::cout << "\nSolving with smoothed aggregation preconditioner and jacobi smoother" << std::endl;
 *
 *      cusp::precond::aggregation::smoothed_aggregation<IndexType, ValueType, MemorySpace> M(A);
 *
 *      // print AMG statistics
 *      M.print();
 *
 *      // allocate storage for solution (x) and right hand side (b)
 *      cusp::array1d<ValueType, MemorySpace> x(A.num_rows, 0);
 *      cusp::array1d<ValueType, MemorySpace> b(A.num_rows, 1);
 *
 *      // set stopping criteria (iteration_limit = 1000, relative_tolerance = 1e-10)
 *      cusp::monitor<ValueType> monitor(b, 1000, 1e-10);
 *
 *      // solve
 *      cusp::krylov::cg(A, x, b, monitor, M);
 *
 *      // report status
 *      monitor.print();
 *
 *      return 0;
 *  }
 *  \endcode
 */
template <typename IndexType,
          typename ValueType,
          typename MemorySpace,
	        typename SmootherType = thrust::use_default,
	        typename SolverType   = thrust::use_default,
          typename Format       = thrust::use_default>
class smoothed_aggregation :
	public cusp::multilevel<IndexType,ValueType,MemorySpace,Format,SmootherType,SolverType>::container
{
  private:

    typedef typename select_sa_matrix_type<IndexType,ValueType,MemorySpace>::type								  SetupMatrixType;
    typedef typename cusp::multilevel<IndexType,ValueType,MemorySpace,Format,SmootherType,SolverType>::container  ML;

  public:

    typedef smoothed_aggregation_options<IndexType,ValueType,MemorySpace>                      SAOptionsType;
    const SAOptionsType & sa_options;
    std::vector< sa_level<SetupMatrixType> > sa_levels;

    smoothed_aggregation(void)
      : sa_options(SAOptionsType()) {}

    template <typename MatrixType>
    smoothed_aggregation(const MatrixType& A, const SAOptionsType& sa_options = SAOptionsType());

    template <typename MatrixType,typename ArrayType>
    smoothed_aggregation(const MatrixType& A, const ArrayType& B, const SAOptionsType& sa_options = SAOptionsType(),
                         typename thrust::detail::enable_if_convertible<typename ArrayType::format,cusp::array1d_format>::type* = 0);

    template <typename MemorySpace2,typename SmootherType2,typename SolverType2,typename Format2>
    smoothed_aggregation(const smoothed_aggregation<IndexType,ValueType,MemorySpace2,SmootherType2,SolverType2,Format2>& M);

    template <typename MatrixType>
    void sa_initialize(const MatrixType& A);

    template <typename MatrixType, typename ArrayType>
    void sa_initialize(const MatrixType& A, const ArrayType& B);

protected:

    template <typename MatrixType>
    void extend_hierarchy(const MatrixType& A);
};
/*! \}
 */

} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

#include <cusp/precond/aggregation/detail/smoothed_aggregation.inl>

