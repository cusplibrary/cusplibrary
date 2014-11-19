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

/*! \file jacobi.h
 *  \brief Jacobi relaxation.
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
 * \brief Represents a Jacobi relaxation scheme
 *
 * \tparam ValueType value_type of the array
 * \tparam MemorySpace memory space of the array (\c cusp::host_memory or \c cusp::device_memory)
 *
 * \par Overview
 * Extracts the matrix diagonal and performs weighted Jacobi relaxation
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/csr_matrix.h>
 * #include <cusp/monitor.h>
 *
 * #incldue <cusp/gallery/poisson.h>
 * #incldue <cusp/monitor/cg.h>
 *
 * // include cusp jacobi header file
 * #include <cusp/relaxation/jacobi.h>
 *
 * int main()
 * {
 *    cusp::csr_matrix<int, float, cusp::device_memory> A;
 *
 *    cusp::gallery::poisson5pt(A, 5, 5);
 *
 *    cusp::relaxation::jacobi<float, cusp::device_memory> M(A, 4.0/3.0);
 *
 *    cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0);
 *    cusp::array1d<float, cusp::device_memory> b(A.num_rows, 1);
 *
 *    cusp::monitor<float> M(b, 20, 1e-4, 0, true);
 *
 *    cusp::krylov::cg(A, x, b, monitor, M);
 * }
 */
template <typename ValueType, typename MemorySpace>
class jacobi : public cusp::linear_operator<ValueType, MemorySpace>
{
public:
    ValueType default_omega;
    cusp::array1d<ValueType,MemorySpace> diagonal;
    cusp::array1d<ValueType,MemorySpace> temp;

    // constructor
    jacobi(void) : default_omega(0.0) {}

    template <typename MatrixType>
    jacobi(const MatrixType& A, ValueType omega=1.0);

    template<typename MemorySpace2>
    jacobi(const jacobi<ValueType,MemorySpace2>& A)
        : default_omega(A.default_omega), temp(A.temp), diagonal(A.diagonal){}

    template <typename MatrixType>
    jacobi(const cusp::precond::aggregation::sa_level<MatrixType>& sa_level, ValueType weight=4.0/3.0);

    // ignores initial x
    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void presmooth(const MatrixType& A, const VectorType1& b, VectorType2& x);

    // smooths initial x
    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void postsmooth(const MatrixType& A, const VectorType1& b, VectorType2& x);

    template <typename MatrixType, typename VectorType1, typename VectorType2>
    void operator()(const MatrixType& A, const VectorType1& b, VectorType2& x);

    template <typename MatrixType, typename VectorType1, typename VectorType2>
    void operator()(const MatrixType& A, const VectorType1& b, VectorType2& x, ValueType omega);
};

} // end namespace relaxation
} // end namespace cusp

#include <cusp/relaxation/detail/jacobi.inl>

