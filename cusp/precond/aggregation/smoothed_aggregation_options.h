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

} // end namespace detail

template<typename ValueType>
class smoothed_aggregation_options
{

public:

    const ValueType theta;
    const ValueType omega;
    const size_t min_level_size;
    const size_t max_levels;

    smoothed_aggregation_options(const ValueType theta = 0.0, const ValueType omega = 4.0/3.0,
                                 const size_t coarse_grid_size = 100, const size_t max_levels = 20)
        : theta(theta), omega(omega), min_level_size(coarse_grid_size), max_levels(max_levels)
    {}

    template <typename MatrixType>
    double estimate_rho_Dinv_A(const MatrixType& A)
    {
        detail::Dinv_A<MatrixType> Dinv_A(A);

        return cusp::detail::ritz_spectral_radius(Dinv_A, 8);
    }

    template<typename MatrixType>
    void strength_of_connection(const MatrixType& A, MatrixType& C)
    {
        cusp::precond::aggregation::symmetric_strength_of_connection(A, C, theta);
    }

    template<typename MatrixType, typename ArrayType>
    void aggregate(const MatrixType& C, ArrayType& aggregates)
    {
        cusp::precond::aggregation::standard_aggregation(C, aggregates);
    }

    template<typename ArrayType1, typename ArrayType2, typename MatrixType>
    void fit_candidates(const ArrayType1& aggregates, const ArrayType2& B, MatrixType& T, ArrayType2& B_coarse)
    {
        cusp::precond::aggregation::fit_candidates(aggregates, B, T, B_coarse);
    }

    template<typename MatrixType>
    void smooth_prolongator(const MatrixType& A, const MatrixType& T, MatrixType& P, ValueType& rho_DinvA)
    {
        // compute spectral radius of diag(C)^-1 * C
        rho_DinvA = estimate_rho_Dinv_A(A);

        cusp::precond::aggregation::smooth_prolongator(A, T, P, omega, rho_DinvA);
    }

    template<typename MatrixType>
    void form_restriction(const MatrixType& P, MatrixType& R)
    {
        cusp::transpose(P,R);
    }

    template<typename MatrixType>
    void galerkin_product(const MatrixType& R, const MatrixType& A, const MatrixType& P, MatrixType& RAP)
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
