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

/*! \file lapack.inl
 *  \brief Two-dimensional array
 */

#pragma once

#include <cublas_v2.h>

namespace cusp
{
namespace blas
{
namespace cublas
{
namespace detail
{

cublasStatus_t axpy( cublasHandle_t handle, int n, float* alpha, float* x, int incx, float* y, int incy)
{
  return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}
cublasStatus_t axpy( cublasHandle_t handle, int n, double* alpha, double* x, int incx, double* y, int incy)
{
  return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t gemv( cublasHandle_t handle, cublasOperation_t trans,
                      int m, int n, float* alpha, float* A, int lda,
                      float* x, int incx, float* beta, float* y, int incy)
{
  return cublasSgemv(handle, trans, m, n,
                     alpha, A, lda, x, incx, beta, y, incy);
}
cublasStatus_t gemv( cublasHandle_t handle, cublasOperation_t trans,
                      int m, int n, double* alpha, double* A, int lda,
                      double* x, int incx, double* beta, double* y, int incy)
{
  return cublasDgemv(handle, trans, m, n,
                     alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t gemm( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                      int m, int n, int k, float* alpha, float* A, int lda,
                      float* B, int ldb, float* beta, float* C, int ldc)
{
  return cublasSgemm(handle, transa, transb, m, n, k,
                     alpha, A, lda, B, ldb, beta, C, ldc);
}
cublasStatus_t gemm( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                      int m, int n, int k, double* alpha, double* A, int lda,
                      double* B, int ldb, double* beta, double* C, int ldc)
{
  return cublasDgemm(handle, transa, transb, m, n, k,
                     alpha, A, lda, B, ldb, beta, C, ldc);
}

} // end namespace detail
} // end namespace cublas
} // end namespace blas
} // end namespace cusp
