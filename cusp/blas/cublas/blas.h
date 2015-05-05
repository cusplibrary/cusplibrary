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
    int lda = A.pitch;

    ValueType alpha = 1.0;
    ValueType beta = 0.0;

    const ValueType *A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType *x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType *y_p = thrust::raw_pointer_cast(&y[0]);

    cublasStatus_t result =
        cublas::detail::gemv(exec.get_handle(), trans, m, n, alpha,
                             A_p, lda, x_p, 1, beta, y_p, 1);

    if(result != CUBLAS_STATUS_SUCCESS)
        throw cusp::runtime_exception("CUBLAS gemv failed!");
}

template<typename Array1d1,
         typename Array1d2,
         typename Array2d1>
void ger(cublas::execution_policy& exec,
         const Array1d1& x,
         const Array1d2& y,
               Array2d1& A)
{
    typedef typename Array2d1::value_type ValueType;

    int m = A.num_rows;
    int n = A.num_cols;
    int lda = A.pitch;

    ValueType alpha = 1.0;

    const ValueType *x_p = thrust::raw_pointer_cast(&x[0]);
    const ValueType *y_p = thrust::raw_pointer_cast(&y[0]);
    ValueType *A_p = thrust::raw_pointer_cast(&A(0,0));

    cublasStatus_t result =
        cublas::detail::ger(exec.get_handle(), m, n, alpha,
                            x_p, 1, y_p, 1, A_p, lda);

    if(result != CUBLAS_STATUS_SUCCESS)
        throw cusp::runtime_exception("CUBLAS gemv failed!");
}

template<typename Array2d1,
         typename Array1d1,
         typename Array1d2>
void symv(cublas::execution_policy& exec,
          const Array2d1& A,
          const Array1d1& x,
                Array1d2& y)
{
    typedef typename Array2d1::value_type ValueType;

    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    int n = A.num_rows;
    int lda = A.pitch;

    ValueType alpha = 1.0;
    ValueType beta = 0.0;

    const ValueType *A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType *x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType *y_p = thrust::raw_pointer_cast(&y[0]);

    cublasStatus_t result =
        cublas::detail::symv(exec.get_handle(), uplo, n, alpha,
                             A_p, lda, x_p, 1, beta, y_p, 1);

    if(result != CUBLAS_STATUS_SUCCESS)
        throw cusp::runtime_exception("CUBLAS symv failed!");
}

template<typename Array1d,
         typename Array2d>
void syr(cublas::execution_policy& exec,
         const Array1d& x,
               Array2d& A)
{
    typedef typename Array2d::value_type ValueType;

    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    int n = A.num_cols;
    int lda = A.pitch;

    ValueType alpha = 1.0;

    const ValueType *x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType *A_p = thrust::raw_pointer_cast(&A(0,0));

    cublasStatus_t result =
        cublas::detail::syr(exec.get_handle(), uplo, n, alpha,
                            x_p, 1, A_p, lda);

    if(result != CUBLAS_STATUS_SUCCESS)
        throw cusp::runtime_exception("CUBLAS syr failed!");
}

template<typename Array2d,
         typename Array1d>
void trmv(cublas::execution_policy& exec,
          const Array2d& A,
                Array1d& x)
{
    typedef typename Array2d::value_type ValueType;

    cublasFillMode_t  uplo  = CUBLAS_FILL_MODE_UPPER;
    cublasOperation_t trans = CUBLAS_OP_N;
    cublasDiagType_t  diag  = CUBLAS_DIAG_NON_UNIT;

    int n = A.num_rows;
    int lda = A.pitch;

    const ValueType *A_p = thrust::raw_pointer_cast(&A(0,0));
    ValueType *x_p = thrust::raw_pointer_cast(&x[0]);

    cublasStatus_t result =
        cublas::detail::trmv(exec.get_handle(), uplo, trans, diag, n,
                             A_p, lda, x_p, 1);

    if(result != CUBLAS_STATUS_SUCCESS)
        throw cusp::runtime_exception("CUBLAS trmv failed!");
}

template<typename Array2d,
         typename Array1d>
void trsv(cublas::execution_policy& exec,
          const Array2d& A,
                Array1d& x)
{
    typedef typename Array2d::value_type ValueType;

    cublasFillMode_t  uplo  = CUBLAS_FILL_MODE_UPPER;
    cublasOperation_t trans = CUBLAS_OP_N;
    cublasDiagType_t  diag  = CUBLAS_DIAG_NON_UNIT;

    int n = A.num_rows;
    int lda = A.pitch;

    const ValueType *A_p = thrust::raw_pointer_cast(&A(0,0));
    ValueType *x_p = thrust::raw_pointer_cast(&x[0]);

    cublasStatus_t result =
        cublas::detail::trsv(exec.get_handle(), uplo, trans, diag, n,
                             A_p, lda, x_p, 1);

    if(result != CUBLAS_STATUS_SUCCESS)
        throw cusp::runtime_exception("CUBLAS trsv failed!");
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

    cublasStatus_t result =
        cublas::detail::gemm(exec.get_handle(), transa, transb,
                             m, n, k, alpha, A_p, m,
                             B_p, k, beta, C_p, m);

    if(result != CUBLAS_STATUS_SUCCESS)
        throw cusp::runtime_exception("CUBLAS gemm failed!");
}

template<typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void symm(cublas::execution_policy& exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C)
{
    typedef typename Array2d1::value_type ValueType;

    cublasSideMode_t side = CUBLAS_SIDE_LEFT;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    int m = A.num_rows;
    int n = B.num_cols;
    int lda = A.pitch;
    int ldb = B.pitch;
    int ldc = C.pitch;

    ValueType alpha = 1.0;
    ValueType beta = 0.0;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));
    ValueType * C_p = thrust::raw_pointer_cast(&C(0,0));

    cublasStatus_t result =
        cublas::detail::symm(exec.get_handle(), side, uplo,
                             m, n, alpha, A_p, lda,
                             B_p, ldb, beta, C_p, ldc);

    if(result != CUBLAS_STATUS_SUCCESS)
        throw cusp::runtime_exception("CUBLAS symm failed!");
}

template<typename Array2d1,
         typename Array2d2>
void syrk(cublas::execution_policy& exec,
          const Array2d1& A,
                Array2d2& B)
{
    typedef typename Array2d1::value_type ValueType;

    cublasFillMode_t  uplo  = CUBLAS_FILL_MODE_UPPER;
    cublasOperation_t trans = CUBLAS_OP_N;

    int n = A.num_rows;
    int k = A.num_cols;
    int lda = A.pitch;
    int ldb = B.pitch;

    ValueType alpha = 1.0;
    ValueType beta = 0.0;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));

    cublasStatus_t result =
        cublas::detail::syrk(exec.get_handle(), uplo, trans,
                             n, k, alpha, A_p, lda,
                             beta, B_p, ldb);

    if(result != CUBLAS_STATUS_SUCCESS)
        throw cusp::runtime_exception("CUBLAS syrk failed!");
}

template<typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void syr2k(cublas::execution_policy& exec,
           const Array2d1& A,
           const Array2d2& B,
                 Array2d3& C)
{
    typedef typename Array2d1::value_type ValueType;

    cublasFillMode_t  uplo  = CUBLAS_FILL_MODE_UPPER;
    cublasOperation_t trans = CUBLAS_OP_N;

    int n = A.num_rows;
    int k = A.num_cols;
    int lda = A.pitch;
    int ldb = B.pitch;
    int ldc = C.pitch;

    ValueType alpha = 1.0;
    ValueType beta = 0.0;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));
    ValueType * C_p = thrust::raw_pointer_cast(&C(0,0));

    cublasStatus_t result =
        cublas::detail::syr2k(exec.get_handle(), uplo, trans,
                              n, k, alpha, A_p, lda,
                              B_p, ldb, beta, C_p, ldc);

    if(result != CUBLAS_STATUS_SUCCESS)
        throw cusp::runtime_exception("CUBLAS syr2k failed!");
}

template<typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void trmm(cublas::execution_policy& exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C)
{
    typedef typename Array2d1::value_type ValueType;

    cublasSideMode_t  side  = CUBLAS_SIDE_LEFT;
    cublasFillMode_t  uplo  = CUBLAS_FILL_MODE_UPPER;
    cublasOperation_t trans = CUBLAS_OP_N;
    cublasDiagType_t  diag  = CUBLAS_DIAG_NON_UNIT;

    int n = A.num_rows;
    int k = A.num_cols;
    int lda = A.pitch;
    int ldb = B.pitch;
    int ldc = C.pitch;

    ValueType alpha = 1.0;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));
    ValueType * C_p = thrust::raw_pointer_cast(&C(0,0));

    cublasStatus_t result =
        cublas::detail::trmm(exec.get_handle(), side, uplo, trans, diag,
                              n, k, alpha, A_p, lda,
                              B_p, ldb, C_p, ldc);

    if(result != CUBLAS_STATUS_SUCCESS)
        throw cusp::runtime_exception("CUBLAS trmm failed!");
}

template<typename Array2d1,
         typename Array2d2>
void trsm(cublas::execution_policy& exec,
          const Array2d1& A,
                Array2d2& B)
{
    typedef typename Array2d1::value_type ValueType;

    cublasSideMode_t  side  = CUBLAS_SIDE_LEFT;
    cublasFillMode_t  uplo  = CUBLAS_FILL_MODE_UPPER;
    cublasOperation_t trans = CUBLAS_OP_N;
    cublasDiagType_t  diag  = CUBLAS_DIAG_NON_UNIT;

    int n = A.num_rows;
    int k = A.num_cols;
    int lda = A.pitch;
    int ldb = B.pitch;

    ValueType alpha = 1.0;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));

    cublasStatus_t result =
        cublas::detail::trsm(exec.get_handle(), side, uplo, trans, diag,
                              n, k, alpha, A_p, lda, B_p, ldb);

    if(result != CUBLAS_STATUS_SUCCESS)
        throw cusp::runtime_exception("CUBLAS trsm failed!");
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

    if(cublas::detail::asum(exec.get_handle(), n, x_p, 1, result) != CUBLAS_STATUS_SUCCESS)
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

    if(cublas::detail::amax(exec.get_handle(), n, x_p, 1, index) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS amax failed!");
    }

    return x[index-1];
}

} // end namespace cublas
} // end namespace blas
} // end namespace cusp

