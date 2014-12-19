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

static cublasStatus_t axpy( cublasHandle_t handle, int n, float* alpha, const float* x, int incx, float* y, int incy )
{
  return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}
static cublasStatus_t axpy( cublasHandle_t handle, int n, double* alpha, const double* x, int incx, double* y, int incy )
{
  return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}
static cublasStatus_t axpy( cublasHandle_t handle, int n, cusp::complex<float>* alpha,
                     const cusp::complex<float>* x, int incx,
                     cusp::complex<float>* y, int incy )
{
  return cublasCaxpy(handle, n, (cuComplex*) alpha, (const cuComplex*) x, incx, (cuComplex*) y, incy);
}
static cublasStatus_t axpy( cublasHandle_t handle, int n, cusp::complex<double>* alpha,
                     const cusp::complex<double>* x, int incx,
                     cusp::complex<double>* y, int incy )
{
  return cublasZaxpy(handle, n, (cuDoubleComplex*) alpha, (const cuDoubleComplex*) x, incx, (cuDoubleComplex*) y, incy);
}

static cublasStatus_t dot( cublasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result )
{
  return cublasSdot(handle, n, x, incx, y, incy, result);
}
static cublasStatus_t dot( cublasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result )
{
  return cublasDdot(handle, n, x, incx, y, incy, result);
}
static cublasStatus_t dotc( cublasHandle_t handle, int n, const float* x, int incx,
                     const float* y, int incy, float* result )
{
  return cublasSdot(handle, n, x, incx, y, incy, result);
}
static cublasStatus_t dotc( cublasHandle_t handle, int n, const double* x, int incx,
                     const double* y, int incy, double* result )
{
  return cublasDdot(handle, n, x, incx, y, incy, result);
}
static cublasStatus_t dotc( cublasHandle_t handle, int n, const cusp::complex<float>* x, int incx,
                     const cusp::complex<float>* y, int incy, cusp::complex<float>* result )
{
  return cublasCdotu(handle, n, (const cuComplex*) x, incx, (const cuComplex*) y, incy, (cuComplex*) result);
}
static cublasStatus_t dotc( cublasHandle_t handle, int n, const cusp::complex<double>* x, int incx,
                     const cusp::complex<double>* y, int incy, cusp::complex<double>* result )
{
  return cublasZdotu(handle, n, (const cuDoubleComplex*) x, incx, (const cuDoubleComplex*) y, incy, (cuDoubleComplex*) result);
}

static cublasStatus_t asum( cublasHandle_t handle, int n, const float* x, int incx, float* result )
{
  return cublasSasum(handle, n, x, incx, result);
}
static cublasStatus_t asum( cublasHandle_t handle, int n, const double* x, int incx, double* result )
{
  return cublasDasum(handle, n, x, incx, result);
}
static cublasStatus_t asum( cublasHandle_t handle, int n, const cusp::complex<float>* x, int incx, float* result )
{
  return cublasScasum(handle, n, (const cuComplex*) x, incx, result);
}
static cublasStatus_t asum( cublasHandle_t handle, int n, const cusp::complex<double>* x, int incx, double* result )
{
  return cublasDzasum(handle, n, (const cuDoubleComplex*) x, incx, result);
}

static cublasStatus_t amax( cublasHandle_t handle, int n, const float* x, int incx, int* index )
{
  return cublasIsamax(handle, n, x, incx, index);
}
static cublasStatus_t amax( cublasHandle_t handle, int n, const double* x, int incx, int* index )
{
  return cublasIdamax(handle, n, x, incx, index);
}
static cublasStatus_t amax( cublasHandle_t handle, int n, const cusp::complex<float>* x, int incx, int* index )
{
  return cublasIcamax(handle, n, (const cuComplex*) x, incx, index);
}
static cublasStatus_t amax( cublasHandle_t handle, int n, const cusp::complex<double>* x, int incx, int* index )
{
  return cublasIzamax(handle, n, (const cuDoubleComplex*) x, incx, index);
}

static cublasStatus_t nrm2( cublasHandle_t handle, int n, const float* x, int incx, float* result )
{
  return cublasSnrm2(handle, n, x, incx, result);
}
static cublasStatus_t nrm2( cublasHandle_t handle, int n, const double* x, int incx, double* result )
{
  return cublasDnrm2(handle, n, x, incx, result);
}
static cublasStatus_t nrm2( cublasHandle_t handle, int n, const cusp::complex<float>* x, int incx, float* result )
{
  return cublasScnrm2(handle, n, (const cuComplex*) x, incx, result);
}
static cublasStatus_t nrm2( cublasHandle_t handle, int n, const cusp::complex<double>* x, int incx, double* result )
{
  return cublasDznrm2(handle, n, (const cuDoubleComplex*) x, incx, result);
}

static cublasStatus_t scal( cublasHandle_t handle, int n, float* alpha, float* x, int incx )
{
  return cublasSscal(handle, n, alpha, x, incx);
}
static cublasStatus_t scal( cublasHandle_t handle, int n, double* alpha, double* x, int incx )
{
  return cublasDscal(handle, n, alpha, x, incx);
}
static cublasStatus_t scal( cublasHandle_t handle, int n, float* alpha, cusp::complex<float>* x, int incx )
{
  return CUBLAS_STATUS_EXECUTION_FAILED;
  // return cublasCscal(handle, n, alpha, x, incx);
}
static cublasStatus_t scal( cublasHandle_t handle, int n, double* alpha, cusp::complex<double>* x, int incx )
{
  return CUBLAS_STATUS_EXECUTION_FAILED;
  // return cublasZscal(handle, n, alpha, x, incx);
}

static cublasStatus_t gemv( cublasHandle_t handle, cublasOperation_t trans,
                      int m, int n, float* alpha, float* A, int lda,
                      float* x, int incx, float* beta, float* y, int incy )
{
  return cublasSgemv(handle, trans, m, n,
                     alpha, A, lda, x, incx, beta, y, incy);
}
static cublasStatus_t gemv( cublasHandle_t handle, cublasOperation_t trans,
                      int m, int n, double* alpha, double* A, int lda,
                      double* x, int incx, double* beta, double* y, int incy )
{
  return cublasDgemv(handle, trans, m, n,
                     alpha, A, lda, x, incx, beta, y, incy);
}

static cublasStatus_t gemm( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                      int m, int n, int k, float* alpha, const float* A, int lda,
                      const float* B, int ldb, float* beta, float* C, int ldc )
{
  return cublasSgemm(handle, transa, transb, m, n, k,
                     alpha, A, lda, B, ldb, beta, C, ldc);
}
static cublasStatus_t gemm( cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                      int m, int n, int k, double* alpha, const double* A, int lda,
                      const double* B, int ldb, double* beta, double* C, int ldc )
{
  return cublasDgemm(handle, transa, transb, m, n, k,
                     alpha, A, lda, B, ldb, beta, C, ldc);
}

} // end namespace detail
} // end namespace cublas
} // end namespace blas
} // end namespace cusp
