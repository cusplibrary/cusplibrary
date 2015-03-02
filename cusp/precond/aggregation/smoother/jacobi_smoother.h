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

#include <cusp/format_utils.h>
#include <cusp/precond/aggregation/smoothed_aggregation_options.h>
#include <cusp/relaxation/jacobi.h>

#include <thrust/transform.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{

template <typename ValueType>
struct jacobi_presmooth_functor
{
    ValueType omega;

    jacobi_presmooth_functor(ValueType omega) : omega(omega) {}

    __host__ __device__
    ValueType operator()(const ValueType& b, const ValueType& d) const
    {
        return omega * b / d;
    }
};

/*! \addtogroup preconditioners Preconditioners
 *  \ingroup preconditioners
 *  \{
 */

template <typename ValueType, typename MemorySpace>
class jacobi_smoother
{
public:
    size_t num_iters;
    cusp::relaxation::jacobi<ValueType,MemorySpace> M;

    jacobi_smoother(void) {}

    template <typename ValueType2, typename MemorySpace2>
    jacobi_smoother(const jacobi_smoother<ValueType2,MemorySpace2>& A) : num_iters(A.num_iters), M(A.M) {}

    template <typename MatrixType1, typename MatrixType2>
    jacobi_smoother(const MatrixType1& A, const sa_level<MatrixType2>& L, ValueType weight=4.0/3.0)
    {
        initialize(A, L, weight);
    }

    template <typename MatrixType1, typename MatrixType2>
    void initialize(const MatrixType1& A, const sa_level<MatrixType2>& L, ValueType weight=4.0/3.0)
    {
        num_iters = L.num_iters;

        M.temp.resize(A.num_rows);

        if(L.rho_DinvA == ValueType(0))
            M.default_omega = weight / detail::estimate_rho_Dinv_A(A);
        else
            M.default_omega = weight / L.rho_DinvA;

        // extract the main diagonal
        cusp::extract_diagonal(A, M.diagonal);
    }

    // ignores initial x
    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void presmooth(const MatrixType& A, const VectorType1& b, VectorType2& x)
    {
        for(size_t i = 0; i < num_iters; i++)
          // x <- omega * D^-1 * b
          thrust::transform(b.begin(), b.end(), M.diagonal.begin(), x.begin(),
                            jacobi_presmooth_functor<ValueType>(M.default_omega));
    }

    // smooths initial x
    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void postsmooth(const MatrixType& A, const VectorType1& b, VectorType2& x)
    {
        for(size_t i = 0; i < num_iters; i++)
          M(A, b, x);
    }
};
/*! \}
 */

} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

