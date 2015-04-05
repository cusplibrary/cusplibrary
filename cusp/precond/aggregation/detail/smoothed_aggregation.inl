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

#include <cusp/array1d.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType, typename SolverType, typename Format>
template <typename MatrixType>
smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType,SolverType,Format>
::smoothed_aggregation(const MatrixType& A, const SAOptionsType& sa_options)
	: ML(), sa_options(sa_options)
{
    sa_initialize(A);
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType, typename SolverType, typename Format>
template <typename MatrixType, typename ArrayType>
smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType,SolverType,Format>
::smoothed_aggregation(const MatrixType& A, const ArrayType& B, const SAOptionsType& sa_options,
                       typename thrust::detail::enable_if_convertible<typename ArrayType::format,cusp::array1d_format>::type*)
	: ML(), sa_options(sa_options)
{
    sa_initialize(A,B);
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType, typename SolverType, typename Format>
template <typename MemorySpace2, typename SmootherType2, typename SolverType2, typename Format2>
smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType,SolverType,Format>
::smoothed_aggregation(const smoothed_aggregation<IndexType,ValueType,MemorySpace2,SmootherType2,SolverType2,Format2>& M)
    : ML(M), sa_options(M.sa_options)
{
    for( size_t lvl = 0; lvl < M.sa_levels.size(); lvl++ )
        sa_levels.push_back(M.sa_levels[lvl]);
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType, typename SolverType, typename Format>
template <typename MatrixType>
void smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType,SolverType,Format>
::sa_initialize(const MatrixType& A)
{
    cusp::constant_array<ValueType> B(A.num_rows, 1);
    sa_initialize(A,B);
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType, typename SolverType, typename Format>
template <typename MatrixType, typename ArrayType>
void smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType,SolverType,Format>
::sa_initialize(const MatrixType& A, const ArrayType& B)
{
    typedef typename select_sa_matrix_view<MatrixType>::type View;
    typedef typename ML::level Level;

    if(sa_levels.size() > 0)
    {
        sa_levels.resize(0);
        ML::levels.resize(0);
    }

    ML::resize(A.num_rows, A.num_cols, A.num_entries);
    ML::levels.reserve(sa_options.max_levels); // avoid reallocations which force matrix copies
    ML::levels.push_back(Level());

    sa_levels.push_back(sa_level<SetupMatrixType>());
    sa_levels.back().B = B;

    // Setup the first level using a COO view
    if(A.num_rows > sa_options.min_level_size)
    {
        View A_(A);
        extend_hierarchy(A_);
        ML::setup_level(0, A, sa_levels[0]);
    }

    // Iteratively setup lower levels until stopping criteria are reached
    while ((sa_levels.back().A_.num_rows > sa_options.min_level_size) &&
            (sa_levels.size() < sa_options.max_levels))
        extend_hierarchy(sa_levels.back().A_);

    // Setup multilevel arrays and matrices on each level
    for( size_t lvl = 1; lvl < sa_levels.size(); lvl++ )
      ML::setup_level(lvl, sa_levels[lvl].A_, sa_levels[lvl]);

    // Initialize coarse solver
    ML::initialize_coarse_solver();
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType, typename SolverType, typename Format>
template <typename MatrixType>
void smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType,SolverType,Format>
::extend_hierarchy(const MatrixType& A)
{
    typedef typename ML::level Level;

    cusp::array1d<IndexType,MemorySpace> aggregates(A.num_rows, IndexType(0));
    {
        // compute stength of connection matrix
        SetupMatrixType C;
        sa_options.strength_of_connection(A, C);

        // compute aggregates
        sa_options.aggregate(C, aggregates);
    }

    SetupMatrixType P;
    cusp::array1d<ValueType,MemorySpace>  B_coarse;
    {
        // compute tenative prolongator and coarse nullspace vector
        sa_options.fit_candidates(aggregates, sa_levels.back().B, sa_levels.back().T, B_coarse);

        // compute prolongation operator
        sa_options.smooth_prolongator(A, sa_levels.back().T, P, sa_levels.back().rho_DinvA);  // TODO if C != A then compute rho_Dinv_C
    }

    // compute restriction operator (transpose of prolongator)
    SetupMatrixType R;
    sa_options.form_restriction(P,R);

    // construct Galerkin product R*A*P
    SetupMatrixType RAP;
    sa_options.galerkin_product(R,A,P,RAP);

    // Setup components for next level in hierarchy
    sa_levels.back().aggregates.swap(aggregates);
    sa_levels.push_back(sa_level<SetupMatrixType>());
    sa_levels.back().A_.swap(RAP);
    sa_levels.back().B.swap(B_coarse);

    ML::copy_or_swap_matrix( ML::levels.back().R, R );
    ML::copy_or_swap_matrix( ML::levels.back().P, P );
    ML::levels.push_back(Level());
}

} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

