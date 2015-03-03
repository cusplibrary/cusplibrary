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
#include <cusp/precond/aggregation/smoother/jacobi_smoother.h>

#include <vector> // TODO replace with host_vector

namespace cusp
{
namespace precond
{
namespace aggregation
{

/*! \addtogroup preconditioners Preconditioners
 *  \ingroup preconditioners
 *  \{
 */

/*! \p smoothed_aggregation : algebraic multigrid preconditoner based on
 *  smoothed aggregation
 *
 */
template <typename IndexType, typename ValueType, typename MemorySpace,
	  typename SmootherType = jacobi_smoother<ValueType,MemorySpace>,
	  typename SolverType = cusp::detail::lu_solver<ValueType,cusp::host_memory> >
class smoothed_aggregation :
  public cusp::multilevel< typename amg_container<IndexType,ValueType,MemorySpace>::solve_type, SmootherType, SolverType>
{
  private:
    typedef typename amg_container<IndexType,ValueType,MemorySpace>::setup_type SetupMatrixType;
    typedef typename amg_container<IndexType,ValueType,MemorySpace>::solve_type SolveMatrixType;
    typedef smoothed_aggregation_options<IndexType,ValueType,MemorySpace>       SAOptionsType;
    typedef cusp::multilevel<SolveMatrixType,SmootherType,SolverType>           Parent;

  public:

    const SAOptionsType & sa_options;
    std::vector< sa_level<SetupMatrixType> > sa_levels;

    smoothed_aggregation(void)
      : sa_options(SAOptionsType()) {}

    template <typename MatrixType>
    smoothed_aggregation(const MatrixType& A, const SAOptionsType& sa_options = SAOptionsType());

    template <typename MatrixType,typename ArrayType>
    smoothed_aggregation(const MatrixType& A, const ArrayType& B, const SAOptionsType& sa_options = SAOptionsType(),
                         typename thrust::detail::enable_if_convertible<typename ArrayType::format,cusp::array1d_format>::type* = 0);

    template <typename MemorySpace2,typename SmootherType2,typename SolverType2>
    smoothed_aggregation(const smoothed_aggregation<IndexType,ValueType,MemorySpace2,SmootherType2,SolverType2>& M);

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

