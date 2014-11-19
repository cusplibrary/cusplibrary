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

/*! \file polynomial.h
 *  \brief polynomial relaxation.
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/linear_operator.h>

namespace cusp
{
/*! \cond */
namespace precond
{
namespace aggregation
{
// forward definitions
template<typename MatrixType> struct sa_level;
} // end namespace aggregation
} // end namespace precond
/*! \endcond */

namespace relaxation
{

/**
 * \brief Represents a Polynomial relaxation scheme
 *
 * \tparam ValueType value_type of the array
 * \tparam MemorySpace memory space of the array (\c cusp::host_memory or \c cusp::device_memory)
 *
 * \par Overview
 * Performs 3rd degree Polynomial relaxation
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
 * // include cusp polynomial header file
 * #include <cusp/relaxation/polynomial.h>
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
 *    cusp::array1d<float, cusp::host_memory> coefficients;
 *
 *    // Compute spectral radius of A
 *    float rho = cusp::detail::ritz_spectral_radius_symmetric(A, 8);
 *    // Compute 3rd degree Chebyshev polynomial
 *    cusp::relaxation::detail::chebyshev_polynomial_coefficients(rho, coefficients);
 *    // Construct polynomial relaxation class
 *    cusp::relaxation::polynomial<float, cusp::device_memory> M(A, coefficients);
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
class polynomial : public cusp::linear_operator<ValueType, MemorySpace>
{
public:

    // note: default_coefficients lives on the host
    cusp::array1d<ValueType, cusp::host_memory> default_coefficients;
    cusp::array1d<ValueType, MemorySpace> residual;
    cusp::array1d<ValueType, MemorySpace> h;
    cusp::array1d<ValueType, MemorySpace> y;

    polynomial(void){}

    template <typename MatrixType, typename VectorType>
    polynomial(const MatrixType& A, const VectorType& coefficients);

    template<typename MemorySpace2>
    polynomial(const polynomial<ValueType,MemorySpace2>& A)
    : default_coefficients(A.default_coefficients),
      residual(A.residual), h(A.h), y(A.y) {}

    template <typename MatrixType>
    polynomial(const cusp::precond::aggregation::sa_level<MatrixType>& sa_level);

    // ignores initial x
    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void presmooth(const MatrixType& A, const VectorType1& b, VectorType2& x);

    // smooths initial x
    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void postsmooth(const MatrixType& A, const VectorType1& b, VectorType2& x);

    template <typename MatrixType, typename VectorType1, typename VectorType2>
    void operator()(const MatrixType& A, const VectorType1& b, VectorType2& x);

    template <typename MatrixType, typename VectorType1, typename VectorType2, typename VectorType3>
    void operator()(const MatrixType& A, const VectorType1& b, VectorType2& x, const VectorType3& coeffients);
};

} // end namespace relaxation
} // end namespace cusp

#include <cusp/relaxation/detail/polynomial.inl>

