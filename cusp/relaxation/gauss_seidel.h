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

/*! \file gauss_seidel.h
 *  \brief Gauss-Seidel relaxation.
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/linear_operator.h>

namespace cusp
{
namespace relaxation
{

/* \cond */
typedef enum
{
    FORWARD,
    BACKWARD,
    SYMMETRIC
} sweep;
/* \endcond */

/**
 * \brief Represents a Gauss-Seidel relaxation scheme
 *
 * \tparam ValueType value_type of the array
 * \tparam MemorySpace memory space of the array (\c cusp::host_memory or \c cusp::device_memory)
 *
 * \par Overview
 * Computes vertex coloring and performs indexed Gauss-Seidel relaxation
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/csr_matrix.h>
 * #include <cusp/monitor.h>
 *
 * #include <cusp/blas/blas.h>
 * #include <cusp/linear_operator.h>
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp gauss_seidel header file
 * #include <cusp/relaxation/gauss_seidel.h>
 *
 * int main()
 * {
 *    // Construct 5-pt Poisson example
 *    cusp::csr_matrix<int, float, cusp::device_memory> A;
 *    cusp::gallery::poisson5pt(A, 5, 5);
 *
 *    // Initialize data
 *    cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0);
 *    cusp::array1d<float, cusp::device_memory> b(A.num_rows, 1);
 *
 *    // Allocate temporaries
 *    cusp::array1d<float, cusp::device_memory> r(A.num_rows);
 *
 *    // Construct gauss_seidel relaxation class
 *    cusp::relaxation::gauss_seidel<float, cusp::device_memory> M(A);
 *
 *    // Compute initial residual
 *    cusp::multiply(A, x, r);
 *    cusp::blas::axpy(b, r, float(-1));
 *
 *    // Construct monitor with stopping criteria of 100 iterations or 1e-4 residual error
 *    cusp::monitor<float> monitor(b, 100, 1e-4, 0, true);
 *
 *    // Iteratively solve system
 *    while (!monitor.finished(r))
 *    {
 *        M(A, b, x);
 *        cusp::multiply(A, x, r);
 *        cusp::blas::axpy(b, r, float(-1));
 *        ++monitor;
 *    }
 *  }
 * \endcode
 */
template <typename ValueType, typename MemorySpace>
class gauss_seidel : public cusp::linear_operator<ValueType, MemorySpace>
{
public:

    cusp::array1d<int,MemorySpace> ordering;
    cusp::array1d<int,cusp::host_memory> color_offsets;
    sweep default_direction;

    // constructor
    gauss_seidel(void) {}

    template <typename MatrixType>
    gauss_seidel(const MatrixType& A, sweep default_direction=SYMMETRIC);

    template<typename MemorySpace2>
    gauss_seidel(const gauss_seidel<ValueType,MemorySpace2>& A)
        : ordering(A.ordering), color_offsets(A.color_offsets), default_direction(A.default_direction) {}

    template <typename MatrixType, typename VectorType1, typename VectorType2>
    void operator()(const MatrixType& A, const VectorType1& b, VectorType2& x);

    template <typename MatrixType, typename VectorType1, typename VectorType2>
    void operator()(const MatrixType& A, const VectorType1& b, VectorType2& x, sweep direction);
};

} // end namespace relaxation
} // end namespace cusp

#include <cusp/relaxation/detail/gauss_seidel.inl>

