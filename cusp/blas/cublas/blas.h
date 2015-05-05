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

#include <cusp/blas/cublas/execution_policy.h>
#include <cusp/blas/cublas/stubs.h>

#include <cublas_v2.h>

namespace cusp
{
namespace blas
{
namespace cublas
{

template <typename Array>
int amax(cublas::execution_policy& exec,
         const Array& x)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    int result;

    if(cublas::detail::amax(exec.get_handle(), n, x_p, 1, result) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS amax failed!");
    }

    return result - 1;
}

template <typename Array>
typename Array::value_type
asum(cublas::execution_policy& exec,
     const Array& x)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    ValueType result;

    if(cublas::detail::asum(exec.get_handle(), n, x_p, 1, result) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS asum failed!");
    }

    return result;
}

template <typename Array1,
          typename Array2,
          typename ScalarType>
void axpy(cublas::execution_policy& exec,
          const Array1& x,
                Array2& y,
          const ScalarType alpha)
{
    typedef typename Array2::value_type ValueType;

    int n = y.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    if(cublas::detail::axpy(exec.get_handle(), n, ValueType(alpha), x_p, 1, y_p, 1) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS axpy failed!");
    }
}

template <typename Array1,
          typename Array2>
void copy(cublas::execution_policy& exec,
          const Array1& x,
                Array2& y)
{
    typedef typename Array2::value_type ValueType;

    int n = y.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    if(cublas::detail::copy(exec.get_handle(), n, x_p, 1, y_p, 1) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS copy failed!");
    }
}

template <typename Array1,
          typename Array2>
typename Array1::value_type
dot(cublas::execution_policy& exec,
    const Array1& x,
    const Array2& y)
{
    typedef typename Array2::value_type ValueType;

    int n = y.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    const ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    ValueType result;

    if(cublas::detail::dot(exec.get_handle(), n, x_p, 1, y_p, 1, result) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS dot failed!");
    }

    return result;
}

template <typename Array1,
          typename Array2>
typename Array1::value_type
dotc(cublas::execution_policy& exec,
     const Array1& x,
     const Array2& y)
{
    typedef typename Array2::value_type ValueType;

    int n = y.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    const ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    ValueType result;

    if(cublas::detail::dotc(exec.get_handle(), n, x_p, 1, y_p, 1, &result) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS dotc failed!");
    }

    return result;
}

template <typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm2(cublas::execution_policy& exec,
     const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename cusp::detail::norm_type<ValueType>::type ResultType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    ResultType result;

    if(cublas::detail::nrm2(exec.get_handle(), n, x_p, 1, result) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS nrm2 failed!");
    }

    return result;
}

template <typename Array,
          typename ScalarType>
void scal(cublas::execution_policy& exec,
          Array& x,
          const ScalarType alpha)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    if(cublas::detail::scal(exec.get_handle(), n, alpha, x_p, 1) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS scal failed!");
    }
}

template <typename Array1,
          typename Array2>
void swap(cublas::execution_policy& exec,
          Array1& x,
          Array2& y)
{
    typedef typename Array1::value_type ValueType;

    int n = x.size();

    ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    if(cublas::detail::swap(exec.get_handle(), n, x_p, 1, y_p, 1) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS swap failed!");
    }
}

template<typename Array2d1,
         typename Array1d1,
         typename Array1d2>
void gemv(cublas::execution_policy& exec,
          const Array2d1& A,
          const Array1d1& x,
          Array1d2& y)
{
    typedef typename Array2d1::value_type ValueType;

    cublasOperation_t trans = CUBLAS_OP_N;

    int m = A.num_rows;
    int n = A.num_cols;

    ValueType alpha = 1.0;
    ValueType beta = 0.0;

    const ValueType *A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType *x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType *y_p = thrust::raw_pointer_cast(&y[0]);

    cublasStatus_t result;

    result = cublas::detail::gemv(exec.get_handle(), trans, m, n, &alpha,
                                  A_p, m, x_p, 1, &beta, y_p, 1);

    if(result != CUBLAS_STATUS_SUCCESS)
        throw cusp::runtime_exception("CUBLAS gemv failed!");
}

template<typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void gemm(cublas::execution_policy& exec,
          const Array2d1& A,
          const Array2d2& B,
          Array2d3& C)
{
    typedef typename Array2d1::value_type ValueType;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    int m = A.num_rows;
    int n = B.num_cols;
    int k = B.num_rows;

    ValueType alpha = 1.0;
    ValueType beta = 0.0;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));
    ValueType * C_p = thrust::raw_pointer_cast(&C(0,0));

    cublasStatus_t result;

    result = cublas::detail::gemm(exec.get_handle(), transa, transb,
                                  m, n, k, &alpha, A_p, m,
                                  B_p, k, &beta, C_p, m);

    if(result != CUBLAS_STATUS_SUCCESS)
        throw cusp::runtime_exception("CUBLAS gemm failed!");
}

template <typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm1(cublas::execution_policy& exec,
     const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename cusp::detail::norm_type<ValueType>::type ResultType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    ResultType result;

    if(cublas::detail::asum(exec.get_handle(), n, x_p, 1, &result) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS asum failed!");
    }

    return result;
}

template <typename Array>
typename Array::value_type
nrmmax(cublas::execution_policy& exec,
       const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename cusp::detail::norm_type<ValueType>::type ResultType;

    int index;
    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    if(cublas::detail::amax(exec.get_handle(), n, x_p, 1, &index) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS amax failed!");
    }

    return x[index-1];
}

} // end namespace cublas
} // end namespace blas
} // end namespace cusp

