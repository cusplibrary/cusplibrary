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

#pragma once

#include <cusp/array1d.h>
#include <cusp/exception.h>
#include <cusp/blas/cblas/stubs.h>

namespace cusp
{
namespace blas
{
template <typename Array1,
          typename Array2,
          typename ScalarType>
void axpy(const cblas::detail::blas_policy<typename Array1::memory_space>& policy,
          const Array1& x,
                Array2& y,
          ScalarType alpha)
{
    typedef typename Array1::value_type ValueType;

    int n = y.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    cblas::detail::axpy(n, &alpha, x_p, 1, y_p, 1);
}

template <typename Array1,
         typename Array2>
typename Array1::value_type
dot(const cblas::detail::blas_policy<typename Array1::memory_space>& policy,
    const Array1& x,
    const Array2& y)
{
    typedef typename Array2::value_type ValueType;

    int n = y.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    const ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    return cblas::detail::dot(n, x_p, 1, y_p, 1);
}

template <typename Array1,
         typename Array2>
typename Array1::value_type
dotc(const cblas::detail::blas_policy<typename Array1::memory_space>& policy,
    const Array1& x,
    const Array2& y)
{
    typedef typename Array2::value_type ValueType;

    int n = y.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    const ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    ValueType result;

    cblas::detail::dotc(n, x_p, 1, y_p, 1, &result);

    return result;
}

template <typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm1(const cblas::detail::blas_policy<typename Array::memory_space>& policy,
     const Array& x)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    return cblas::detail::asum(n, x_p, 1);
}

template <typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm2(const cblas::detail::blas_policy<typename Array::memory_space>& policy,
     const Array& x)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    return cblas::detail::nrm2(n, x_p, 1);
}

template <typename Array>
typename Array::value_type
nrmmax(const cblas::detail::blas_policy<typename Array::memory_space>& policy,
       const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename cusp::detail::norm_type<ValueType>::type ResultType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    int index = cblas::detail::amax(n, x_p, 1);

    return x[index];
}

template <typename Array, typename ScalarType>
void scal(const cblas::detail::blas_policy<typename Array::memory_space>& policy,
          Array& x,
          ScalarType alpha)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    cblas::detail::scal(n, &alpha, x_p, 1);
}

template<typename Array2d1, typename Array1d1, typename Array1d2>
void gemv(const cblas::detail::blas_policy<typename Array1d2::memory_space>& policy,
          const Array2d1& A,
          const Array1d1& x,
          Array1d2& y)
{
    typedef typename Array2d1::value_type ValueType;

    enum CBLAS_ORDER order = CblasColMajor;
    enum CBLAS_TRANSPOSE trans = CblasNoTrans;

    int m = A.num_rows;
    int n = A.num_cols;

    ValueType alpha = 1.0;
    ValueType beta = 0.0;

    ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    ValueType * x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType * y_p = thrust::raw_pointer_cast(&y[0]);

    cblas::detail::gemv(order, trans, m, n, alpha,
                 A_p, m, x_p, 1, beta, y_p, 1);
}

template<typename Array2d1, typename Array2d2, typename Array2d3>
void gemm(const cblas::detail::blas_policy<typename Array2d3::memory_space>& policy,
          const Array2d1& A,
          const Array2d2& B,
          Array2d3& C)
{
    typedef typename Array2d1::value_type ValueType;

    enum CBLAS_ORDER order = CblasColMajor;
    enum CBLAS_TRANSPOSE transa = CblasNoTrans;
    enum CBLAS_TRANSPOSE transb = CblasNoTrans;

    int m = A.num_rows;
    int n = B.num_cols;
    int k = B.num_rows;

    ValueType alpha = 1.0;
    ValueType beta = 0.0;

    ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));
    ValueType * C_p = thrust::raw_pointer_cast(&C(0,0));

    cblas::detail::gemm(order, transa, transb,
                 m, n, k, alpha, A_p, m,
                 B_p, k, beta, C_p, m);
}

} // end namespace blas
} // end namespace cusp

