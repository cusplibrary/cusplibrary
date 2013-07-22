/*
 *  Copyright 2008-2009 NVIDIA Corporation
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

#include <cusp/blas.h>
#include <cusp/elementwise.h>
#include <cusp/multiply.h>

#include <cusp/detail/format_utils.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{
namespace detail
{

template <typename Matrix>
void setup_level_matrix(Matrix& dst, Matrix& src) {
    dst.swap(src);
}

template <typename Matrix1, typename Matrix2>
void setup_level_matrix(Matrix1& dst, Matrix2& src) {
    dst = src;
}

} // end namespace detail

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType, typename SolverType>
template <typename MatrixType>
smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType,SolverType>
::smoothed_aggregation(const MatrixType& A)
  : sa_options(default_sa_options)
{
    typedef typename cusp::array1d_view< thrust::constant_iterator<ValueType> > ConstantView;

    ConstantView B(thrust::constant_iterator<ValueType>(1),
                   thrust::constant_iterator<ValueType>(1) + A.num_rows);
    sa_initialize(A, B);
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType, typename SolverType>
template <typename MatrixType, typename Options>
smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType,SolverType>
::smoothed_aggregation(const MatrixType& A,
                       const Options& sa_options)
  : sa_options(sa_options)
{
    typedef typename cusp::array1d_view< thrust::constant_iterator<ValueType> > ConstantView;

    ConstantView B(thrust::constant_iterator<ValueType>(1),
                   thrust::constant_iterator<ValueType>(1) + A.num_rows);
    sa_initialize(A, B);
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType, typename SolverType>
template <typename MatrixType>
smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType,SolverType>
::smoothed_aggregation(const MatrixType& A, const cusp::array1d<ValueType,MemorySpace>& B)
  : sa_options(default_sa_options)
{
    sa_initialize(A, B);
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType, typename SolverType>
template <typename MatrixType, typename Options>
smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType,SolverType>
::smoothed_aggregation(const MatrixType& A, const cusp::array1d<ValueType,MemorySpace>& B,
                       const Options& sa_options)
  : sa_options(sa_options)
{
    sa_initialize(A, B);
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType, typename SolverType>
template <typename MemorySpace2, typename SmootherType2, typename SolverType2>
smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType,SolverType>
::smoothed_aggregation(const smoothed_aggregation<IndexType,ValueType,MemorySpace2,SmootherType2,SolverType2>& M)
    : sa_options(M.sa_options), Parent(M)
{
   for( size_t lvl = 0; lvl < M.sa_levels.size(); lvl++ )
      sa_levels.push_back(M.sa_levels[lvl]);
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType, typename SolverType>
template <typename MatrixType, typename ArrayType>
void smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType,SolverType>
::sa_initialize(const MatrixType& A, const ArrayType& B)
{
    CUSP_PROFILE_SCOPED();

    Parent* ML = this;
    ML->levels.reserve(sa_options.max_levels); // avoid reallocations which force matrix copies

    sa_levels.push_back(sa_level<SetupMatrixType>());
    ML->levels.push_back(typename Parent::level());

    sa_levels.back().B = B;
    sa_levels.back().A_ = A; // copy

    while ((sa_levels.back().A_.num_rows > sa_options.min_level_size) &&
           (sa_levels.size() < sa_options.max_levels))
        extend_hierarchy();

    ML->solver = SolverType(sa_levels.back().A_);

    // Setup solve matrix for each level
    for( size_t lvl = 0; lvl < sa_levels.size(); lvl++ )
        detail::setup_level_matrix( ML->levels[lvl].A, sa_levels[lvl].A_ );
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType, typename SolverType>
void smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType,SolverType>
::extend_hierarchy(void)
{
    CUSP_PROFILE_SCOPED();

    Parent* ML = this;

    cusp::array1d<IndexType,MemorySpace> aggregates;
    {
        // compute stength of connection matrix
        SetupMatrixType C;
        sa_options.strength_of_connection(sa_levels.back().A_, C);

        // compute aggregates
        aggregates.resize(C.num_rows);
        cusp::blas::fill(aggregates,IndexType(0));
        sa_options.aggregate(C, aggregates);
    }

    SetupMatrixType P;
    cusp::array1d<ValueType,MemorySpace>  B_coarse;
    {
        // compute tenative prolongator and coarse nullspace vector
        SetupMatrixType 				T;
        sa_options.fit_candidates(aggregates, sa_levels.back().B, T, B_coarse);

        // compute prolongation operator
        sa_options.smooth_prolongator(sa_levels.back().A_, T, P, sa_levels.back().rho_DinvA);  // TODO if C != A then compute rho_Dinv_C
    }

    // compute restriction operator (transpose of prolongator)
    SetupMatrixType R;
    sa_options.form_restriction(P,R);

    // construct Galerkin product R*A*P
    SetupMatrixType RAP;
    sa_options.galerkin_product(R,sa_levels.back().A_,P,RAP);

    ML->levels.back().smoother = SmootherType(sa_levels.back());

    sa_levels.back().aggregates.swap(aggregates);
    detail::setup_level_matrix( ML->levels.back().R, R );
    detail::setup_level_matrix( ML->levels.back().P, P );
    ML->levels.back().residual.resize(sa_levels.back().A_.num_rows);

    ML->levels.push_back(typename Parent::level());
    sa_levels.push_back(sa_level<SetupMatrixType>());

    sa_levels.back().A_.swap(RAP);
    sa_levels.back().B.swap(B_coarse);
    ML->levels.back().x.resize(sa_levels.back().A_.num_rows);
    ML->levels.back().b.resize(sa_levels.back().A_.num_rows);
}

} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

