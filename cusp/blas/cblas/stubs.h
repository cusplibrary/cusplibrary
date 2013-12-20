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

#include <cblas.h>
#undef complex

namespace cusp
{
namespace blas
{
namespace cblas
{
namespace detail
{

void axpy( int n, float alpha, float* x, int incx, float* y, int incy)
{
  cblas_saxpy(n, alpha, x, incx, y, incy);
}
void axpy( int n, double alpha, double* x, int incx, double* y, int incy)
{
  cblas_daxpy(n, alpha, x, incx, y, incy);
}

void gemv( enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans,
                      int m, int n, float alpha, float* A, int lda,
                      float* x, int incx, float beta, float* y, int incy)
{
  cblas_sgemv(order, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
void gemv( enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans,
                      int m, int n, double alpha, double* A, int lda,
                      double* x, int incx, double beta, double* y, int incy)
{
  cblas_dgemv(order, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

void gemm( enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transa, enum CBLAS_TRANSPOSE transb,
                      int m, int n, int k, float alpha, float* A, int lda,
                      float* B, int ldb, float beta, float* C, int ldc)
{
  cblas_sgemm(order, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
void gemm( enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transa, enum CBLAS_TRANSPOSE transb,
                      int m, int n, int k, double alpha, double* A, int lda,
                      double* B, int ldb, double beta, double* C, int ldc)
{
  cblas_dgemm(order, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

} // end namespace detail
} // end namespace cblas
} // end namespace blas
} // end namespace cusp
