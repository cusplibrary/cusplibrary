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
#include <cusp/monitor.h>
#include <cusp/transpose.h>
#include <cusp/precond/diagonal.h>
#include <cusp/krylov/arnoldi.h>

#include <cusp/graph/maximal_independent_set.h>
#include <cusp/precond/aggregation/aggregate.h>
#include <cusp/precond/aggregation/smooth.h>
#include <cusp/precond/aggregation/strength.h>
#include <cusp/precond/aggregation/tentative.h>

#include <cusp/detail/format_utils.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{
namespace detail
{

template <typename MatrixType>
struct Dinv_A : public cusp::linear_operator<typename MatrixType::value_type, typename MatrixType::memory_space>
{
    const MatrixType& A;
    const cusp::precond::diagonal<typename MatrixType::value_type, typename MatrixType::memory_space> Dinv;

    Dinv_A(const MatrixType& A)
        : A(A), Dinv(A),
          cusp::linear_operator<typename MatrixType::value_type, typename MatrixType::memory_space>(A.num_rows, A.num_cols, A.num_entries + A.num_rows)
    {}

    template <typename Array1, typename Array2>
    void operator()(const Array1& x, Array2& y) const
    {
        cusp::multiply(A,x,y);
        cusp::multiply(Dinv,y,y);
    }
};

template <typename MatrixType>
double estimate_rho_Dinv_A(const MatrixType& A)
{
    typedef typename MatrixType::value_type   ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    Dinv_A<MatrixType> Dinv_A(A);

    return cusp::detail::ritz_spectral_radius(Dinv_A, 8);
}


template <typename Matrix>
void setup_level_matrix(Matrix& dst, Matrix& src) {
    dst.swap(src);
}

template <typename Matrix1, typename Matrix2>
void setup_level_matrix(Matrix1& dst, Matrix2& src) {
    dst = src;
}

} // end namespace detail

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType>
template <typename MatrixType>
smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType>
::smoothed_aggregation(const MatrixType& A, const ValueType theta)
    : theta(theta), Parent()
{
    typedef typename cusp::array1d_view< thrust::constant_iterator<ValueType> > ConstantView;

    ConstantView B(thrust::constant_iterator<ValueType>(1),
                   thrust::constant_iterator<ValueType>(1) + A.num_rows);
    init(A, B);
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType>
template <typename MatrixType, typename ArrayType>
smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType>
::smoothed_aggregation(const MatrixType& A, const ArrayType& B, const ValueType theta)
    : theta(theta), Parent()
{
    init(A, B);
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType>
template <typename MatrixType, typename ArrayType>
void smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType>
::init(const MatrixType& A, const ArrayType& B)
{
    CUSP_PROFILE_SCOPED();

    Parent::levels.reserve(20); // avoid reallocations which force matrix copies

    sa_levels.push_back(sa_level());
    Parent::levels.push_back(typename Parent::level());

    sa_levels.back().B = B;
    sa_levels.back().A_ = A; // copy

    while (sa_levels.back().A_.num_rows > 100)
        extend_hierarchy();

    // TODO make lu_solver accept sparse input
    cusp::array2d<ValueType,cusp::host_memory> coarse_dense(sa_levels.back().A_);
    Parent::LU = cusp::detail::lu_solver<ValueType, cusp::host_memory>(coarse_dense);

    // Setup solve matrix for each level
    Parent::levels[0].A = A;
    for( size_t lvl = 1; lvl < sa_levels.size(); lvl++ )
        detail::setup_level_matrix( Parent::levels[lvl].A, sa_levels[lvl].A_ );
}

template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType>
void smoothed_aggregation<IndexType,ValueType,MemorySpace,SmootherType>
::extend_hierarchy(void)
{
    CUSP_PROFILE_SCOPED();

    cusp::array1d<IndexType,MemorySpace> aggregates;
    {
        // compute stength of connection matrix
        SetupMatrixType C;
        symmetric_strength_of_connection(sa_levels.back().A_, C, theta);

        // compute aggregates
        aggregates.resize(C.num_rows);
        cusp::blas::fill(aggregates,IndexType(0));
        standard_aggregation(C, aggregates);
    }

    // compute spectral radius of diag(C)^-1 * C
    ValueType rho_DinvA = detail::estimate_rho_Dinv_A(sa_levels.back().A_);

    SetupMatrixType P;
    cusp::array1d<ValueType,MemorySpace>  B_coarse;
    {
        // compute tenative prolongator and coarse nullspace vector
        SetupMatrixType 				T;
        fit_candidates(aggregates, sa_levels.back().B, T, B_coarse);

        // compute prolongation operator
        smooth_prolongator(sa_levels.back().A_, T, P, ValueType(4.0/3.0), rho_DinvA);  // TODO if C != A then compute rho_Dinv_C
    }

    // compute restriction operator (transpose of prolongator)
    SetupMatrixType R;
    cusp::transpose(P,R);

    // construct Galerkin product R*A*P
    SetupMatrixType RAP;
    {
        // TODO test speed of R * (A * P) vs. (R * A) * P
        SetupMatrixType AP;
        cusp::multiply(sa_levels.back().A_, P, AP);
        cusp::multiply(R, AP, RAP);
    }

    Parent::levels.back().smoother = smoother_initializer(sa_levels.back().A_, rho_DinvA);

    sa_levels.back().aggregates.swap(aggregates);
    detail::setup_level_matrix( Parent::levels.back().R, R );
    detail::setup_level_matrix( Parent::levels.back().P, P );
    Parent::levels.back().residual.resize(sa_levels.back().A_.num_rows);

    Parent::levels.push_back(typename Parent::level());
    sa_levels.push_back(sa_level());

    sa_levels.back().A_.swap(RAP);
    sa_levels.back().B.swap(B_coarse);
    Parent::levels.back().x.resize(sa_levels.back().A_.num_rows);
    Parent::levels.back().b.resize(sa_levels.back().A_.num_rows);
}

} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

