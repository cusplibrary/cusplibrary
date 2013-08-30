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

#pragma once

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/transpose.h>
#include <cusp/multiply.h>

#include <cusp/krylov/arnoldi.h>

#include <cusp/precond/diagonal.h>
#include <cusp/precond/aggregation/aggregate.h>
#include <cusp/precond/aggregation/smooth.h>
#include <cusp/precond/aggregation/strength.h>
#include <cusp/precond/aggregation/tentative.h>

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
    detail::Dinv_A<MatrixType> Dinv_A(A);

    return cusp::detail::ritz_spectral_radius(Dinv_A, 8);
}


} // end namespace detail

template <typename IndexType, typename ValueType, typename MemorySpace>
struct amg_container {};

template <typename IndexType, typename ValueType>
struct amg_container<IndexType,ValueType,cusp::host_memory>
{
    // use CSR on host
    typedef typename cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> setup_type;
    typedef typename cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> solve_type;
};

template <typename IndexType, typename ValueType>
struct amg_container<IndexType,ValueType,cusp::device_memory>
{
    // use COO on device
    typedef typename cusp::coo_matrix<IndexType,ValueType,cusp::device_memory> setup_type;
    typedef typename cusp::hyb_matrix<IndexType,ValueType,cusp::device_memory> solve_type;
};

template<typename IndexType, typename ValueType, typename MemorySpace>
class smoothed_aggregation_options
{
public:

    typedef typename amg_container<IndexType,ValueType,MemorySpace>::setup_type MatrixType;
    typedef cusp::array1d<IndexType,MemorySpace> IndexArray;
    typedef cusp::array1d<ValueType,MemorySpace> ValueArray;

    const ValueType theta;
    const ValueType omega;
    const size_t min_level_size;
    const size_t max_levels;

    smoothed_aggregation_options(const ValueType theta = 0.0, const ValueType omega = 4.0/3.0,
                                 const size_t min_level_size = 100, const size_t max_levels = 20)
        : theta(theta), omega(omega), min_level_size(min_level_size), max_levels(max_levels)
    {}

    template<typename MemorySpace2>
    smoothed_aggregation_options(const smoothed_aggregation_options<IndexType,ValueType,MemorySpace2>& M)
        : theta(M.theta), omega(M.omega), min_level_size(M.min_level_size), max_levels(M.max_levels)
    {}

    virtual void strength_of_connection(const MatrixType& A, MatrixType& C) const
    {
        cusp::precond::aggregation::symmetric_strength_of_connection(A, C, theta);
    }

    virtual void aggregate(const MatrixType& C, IndexArray& aggregates) const
    {
        cusp::precond::aggregation::standard_aggregation(C, aggregates);
    }

    virtual void fit_candidates(const IndexArray& aggregates, const ValueArray& B, MatrixType& T, ValueArray& B_coarse) const
    {
        cusp::precond::aggregation::fit_candidates(aggregates, B, T, B_coarse);
    }

    virtual void smooth_prolongator(const MatrixType& A, const MatrixType& T, MatrixType& P, ValueType& rho_DinvA) const
    {
        // compute spectral radius of diag(C)^-1 * C
        rho_DinvA = detail::estimate_rho_Dinv_A(A);

        cusp::precond::aggregation::smooth_prolongator(A, T, P, omega, rho_DinvA);
    }

    virtual void form_restriction(const MatrixType& P, MatrixType& R) const
    {
        cusp::transpose(P,R);
    }

    virtual void galerkin_product(const MatrixType& R, const MatrixType& A, const MatrixType& P, MatrixType& RAP) const
    {
        // TODO test speed of R * (A * P) vs. (R * A) * P
        MatrixType AP;
        cusp::multiply(A, P, AP);
        cusp::multiply(R, AP, RAP);
    }
};

} // end namespace aggregation
} // end namespace precond
} // end namespace cusp
