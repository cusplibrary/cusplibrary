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

/*! \file stubs.h
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

static void axpy( int n, float * alpha, const float* x, int incx, float* y, int incy )
{
    cblas_saxpy(n, *alpha, x, incx, y, incy);
}
static void axpy( int n, double * alpha, const double* x, int incx, double* y, int incy )
{
    cblas_daxpy(n, *alpha, x, incx, y, incy);
}
static void axpy( int n, void * alpha, const cusp::complex<float>* x, int incx, cusp::complex<float>* y, int incy )
{
    cblas_caxpy(n, alpha, (void*)x, incx, (void*)y, incy);
}
static void axpy( int n, void * alpha, const cusp::complex<double>* x, int incx, cusp::complex<double>* y, int incy )
{
    cblas_zaxpy(n, alpha, (void*)x, incx, (void*)y, incy);
}

static float dot( int n, const float* x, int incx, const float* y, int incy )
{
  return cblas_sdot(n, x, incx, y, incy);
}
static double dot( int n, const double* x, int incx, const double* y, int incy )
{
  return cblas_ddot(n, x, incx, y, incy);
}

static void dotc( int n, const float* x, int incx, const float* y, int incy, float* result )
{
  *result = cblas_sdot(n, x, incx, y, incy);
}
static void dotc( int n, const double* x, int incx, const double* y, int incy, double* result )
{
  *result = cblas_ddot(n, x, incx, y, incy);
}
static void dotc( int n, const cusp::complex<float>* x, int incx, const cusp::complex<float>* y, int incy, cusp::complex<float>* result )
{
  cblas_cdotc_sub(n, (void*)x, incx, (void*)y, incy, (void*)result);
}
static void dotc( int n, const cusp::complex<double>* x, int incx, const cusp::complex<double>* y, int incy, cusp::complex<double>* result )
{
  cblas_zdotc_sub(n, (void*)x, incx, (void*)y, incy, (void*)result);
}

static float asum( int n, const float* x, int incx )
{
  return cblas_sasum(n, x, incx);
}
static double asum( int n, const double* x, int incx )
{
  return cblas_dasum(n, x, incx);
}
static float asum( int n, const cusp::complex<float>* x, int incx )
{
  return cblas_scasum(n, (void*)x, incx);
}
static double asum( int n, const cusp::complex<double>* x, int incx )
{
  return cblas_dzasum(n, (void*)x, incx);
}

static int amax( int n, const float* x, int incx )
{
  return cblas_isamax(n, x, incx);
}
static int amax( int n, const double* x, int incx )
{
  return cblas_idamax(n, x, incx);
}
static int amax( int n, const cusp::complex<float>* x, int incx )
{
  return cblas_icamax(n, (void*)x, incx);
}
static int amax( int n, const cusp::complex<double>* x, int incx )
{
  return cblas_izamax(n, (void*)x, incx);
}

static float nrm2( int n, const float* x, int incx )
{
  return cblas_snrm2(n, x, incx);
}
static double nrm2( int n, const double* x, int incx )
{
  return cblas_dnrm2(n, x, incx);
}
static float nrm2( int n, const cusp::complex<float>* x, int incx )
{
  return cblas_scnrm2(n, (void*)x, incx);
}
static double nrm2( int n, const cusp::complex<double>* x, int incx )
{
  return cblas_dznrm2(n, (void*)x, incx);
}

static void scal( int n, float * alpha, float* x, int incx )
{
  return cblas_sscal(n, *alpha, x, incx);
}
static void scal( int n, double * alpha, double* x, int incx )
{
  return cblas_dscal(n, *alpha, x, incx);
}
static void scal( int n, float * alpha, cusp::complex<float>* x, int incx )
{
  return cblas_cscal(n, alpha, (void*)x, incx);
}
static void scal( int n, double * alpha, cusp::complex<double>* x, int incx )
{
  return cblas_zscal(n, alpha, (void*)x, incx);
}

static void gemv( enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans,
           int m, int n, float alpha, float* A, int lda,
           float* x, int incx, float beta, float* y, int incy)
{
    cblas_sgemv(order, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
static void gemv( enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans,
           int m, int n, double alpha, double* A, int lda,
           double* x, int incx, double beta, double* y, int incy)
{
    cblas_dgemv(order, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

static void gemm( enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transa, enum CBLAS_TRANSPOSE transb,
           int m, int n, int k, float alpha, float* A, int lda,
           float* B, int ldb, float beta, float* C, int ldc)
{
    cblas_sgemm(order, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
static void gemm( enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE transa, enum CBLAS_TRANSPOSE transb,
           int m, int n, int k, double alpha, double* A, int lda,
           double* B, int ldb, double beta, double* C, int ldc)
{
    cblas_dgemm(order, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

} // end namespace detail
} // end namespace cblas
} // end namespace blas
} // end namespace cusp
