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

#include <cusp/complex.h>
#include <cblas.h>

#define CUSP_CBLAS_EXPAND_REAL_DEFS(FUNC_MACRO)               \
  FUNC_MACRO(float , float , s)                               \
  FUNC_MACRO(double, double, d)

#define CUSP_CBLAS_EXPAND_COMPLEX_DEFS_1(FUNC_MACRO)          \
  FUNC_MACRO(cusp::complex<float> , float,  c)                \
  FUNC_MACRO(cusp::complex<double>, double, z)

#define CUSP_CBLAS_EXPAND_COMPLEX_DEFS_2(FUNC_MACRO)          \
  FUNC_MACRO(cusp::complex<float> , float,  sc)               \
  FUNC_MACRO(cusp::complex<double>, double, dz)

#define CUSP_CBLAS_EXPAND_DEFS_1(FUNC_MACRO)                  \
  CUSP_CBLAS_EXPAND_REAL_DEFS(FUNC_MACRO)                     \
  // CUSP_CBLAS_EXPAND_COMPLEX_DEFS_1(FUNC_MACRO)

#define CUSP_CBLAS_EXPAND_DEFS_2(FUNC_MACRO)                  \
  CUSP_CBLAS_EXPAND_REAL_DEFS(FUNC_MACRO)                     \
  // CUSP_CBLAS_EXPAND_COMPLEX_DEFS_2(FUNC_MACRO)

#define CUSP_CBLAS_AMAX(T,V,name)                                                           \
  int amax( const int n, const T* X, const int incX )                                       \
  {                                                                                         \
    return cblas_i##name##amax(n, (const V*) X, incX);                                      \
  }

#define CUSP_CBLAS_ASUM(T,V,name)                                                           \
  V asum( const int n, const T* X, const int incX )                                         \
  {                                                                                         \
    return cblas_##name##asum(n, (const V*) X, incX);                                       \
  }

#define CUSP_CBLAS_AXPY(T,V,name)                                                           \
  void axpy( const int n, const V alpha, const T* X, const int incX, T* Y, const int incY ) \
  {                                                                                         \
    cblas_##name##axpy(n, alpha, (const V*) X, incX, (V*) Y, incY);                         \
  }

#define CUSP_CBLAS_COPY(T,V,name)                                                           \
  void copy( const int n, const T* X, const int incX, T* Y, const int incY )                \
  {                                                                                         \
    cblas_##name##copy(n, (const V*) X, incX, (V*) Y, incY);                                \
  }

#define CUSP_CBLAS_DOT(T,V,name)                                                            \
  T dot( const int n, const T* X, const int incX, const T* Y, const int incY )              \
  {                                                                                         \
    return cblas_##name##dot(n, (const V*) X, incX, (const V*) Y, incY);                    \
  }

#define CUSP_CBLAS_DOTC(T,V,name)                                                           \
  void dotc( const int n, const T* X, const int incX, const T* Y, const int incY, T* ret )  \
  {                                                                                         \
    cblas_##name##dotc_sub(n, (const V*) X, incX, (const V*) Y, incY, (V*) ret);            \
  }

#define CUSP_CBLAS_DOTU(T,V,name)                                                           \
  void dotu( const int n, const T* X, const int incX, const T* Y, const int incY, T* ret )  \
  {                                                                                         \
    cblas_##name##dotu_sub(n, (const V*) X, incX, (const V*) Y, incY, (V*) ret);            \
  }

#define CUSP_CBLAS_NRM2(T,V,name)                                                           \
  T nrm2( const int n, const T* X, const int incX )                                         \
  {                                                                                         \
    return cblas_##name##nrm2(n, (const V*) X, incX);                                       \
  }

#define CUSP_CBLAS_SCAL(T,V,name)                                                           \
  void scal( const int n, const T alpha, T* X, const int incX )                             \
  {                                                                                         \
    cblas_##name##scal(n, alpha, (V*) X, incX);                                             \
  }

#define CUSP_CBLAS_SWAP(T,V,name)                                                           \
  void swap( const int n, T* X, const int incX, T* Y, const int incY )                      \
  {                                                                                         \
    cblas_##name##swap(n, (V*) X, incX, (V*) Y, incY);                                      \
  }

#define CUSP_CBLAS_GEMV(T,V,name)                                                           \
void gemv( enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans,                              \
           int m, int n, T alpha, const T* A, int lda,                                      \
           const T* x, int incx, T beta, T* y, int incy)                                    \
{                                                                                           \
    cblas_##name##gemv(order, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);          \
}

#define CUSP_CBLAS_GER(T,V,name)                                                            \
  void ger( enum CBLAS_ORDER order, int m, int n, T alpha, const T* x, int incx,            \
            const T* y, int incy, T* A, int lda)                                            \
{                                                                                           \
    return cblas_##name##ger(order, m, n, alpha,                                            \
                             (const V*) x, incx, (const V*) y, incy,                        \
                             (V*) A, lda);                                                  \
}

#define CUSP_CBLAS_SYMV(T,V,name)                                                           \
  void symv( enum CBLAS_ORDER order, enum CBLAS_UPLO uplo,                                  \
             int n, T alpha, const T* A, int lda,                                           \
             const T* x, int incx, T beta, T* y, int incy)                                  \
{                                                                                           \
    return cblas_##name##symv(order, uplo, n, alpha, (const V*) A, lda,                     \
                              (const V*) x, incx, beta, (V*) y, incy);                      \
}

#define CUSP_CBLAS_SYR(T,V,name)                                                            \
  void syr( enum CBLAS_ORDER order, enum CBLAS_UPLO uplo,                                   \
            int n, T alpha, const T* x, int incx, T* A, int lda)                            \
{                                                                                           \
    return cblas_##name##syr(order, uplo, n, alpha,                                         \
                             (const V*) x, incx, (V*) A, lda);                              \
}

#define CUSP_CBLAS_TRMV(T,V,name)                                                           \
  void trmv( enum CBLAS_ORDER order, enum CBLAS_UPLO uplo,                                  \
             enum CBLAS_TRANSPOSE trans, enum CBLAS_DIAG diag,                              \
             int n, const T* A, int lda, const T* x, int incx)                              \
{                                                                                           \
    return cblas_##name##trmv(order, uplo, trans, diag, n,                                  \
                              (const V*) A, lda, (V*) x, incx);                             \
}

#define CUSP_CBLAS_TRSV(T,V,name)                                                           \
  void trsv( enum CBLAS_ORDER order, enum CBLAS_UPLO uplo,                                  \
             enum CBLAS_TRANSPOSE trans, enum CBLAS_DIAG diag,                              \
             int n, const T* A, int lda, const T* x, int incx)                              \
{                                                                                           \
    return cblas_##name##trsv(order, uplo, trans, diag, n,                                  \
                              (const V*) A, lda, (V*) x, incx);                             \
}

#define CUSP_CBLAS_GEMM(T,V,name)                                                           \
void gemm( enum CBLAS_ORDER order,                                                          \
           enum CBLAS_TRANSPOSE transa, enum CBLAS_TRANSPOSE transb,                        \
           int m, int n, int k, T alpha, const T* A, int lda,                               \
           const T* B, int ldb, T beta, T* C, int ldc)                                      \
{                                                                                           \
    cblas_##name##gemm(order, transa, transb,                                               \
                       m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);                       \
}

#define CUSP_CBLAS_SYMM(T,V,name)                                                           \
  void symm( enum CBLAS_ORDER order,                                                        \
                       enum CBLAS_SIDE side, enum CBLAS_UPLO uplo,                          \
                       int m, int n, T alpha, const T* A, int lda,                          \
                       const T* B, int ldb, T beta, T* C, int ldc)                          \
{                                                                                           \
    return cblas_##name##symm(order, side, uplo, m, n,                                      \
                              (V) alpha, (const V*) A, lda, (const V*) B, ldb,              \
                              (V) beta, (V*) C, ldc);                                       \
}

#define CUSP_CBLAS_SYRK(T,V,name)                                                           \
  void syrk( enum CBLAS_ORDER order,                                                        \
                       enum CBLAS_UPLO uplo, enum CBLAS_TRANSPOSE trans,                    \
                       int n, int k, T alpha, const T* A, int lda,                          \
                       T beta, T* C, int ldc)                                               \
{                                                                                           \
    return cblas_##name##syrk(order, uplo, trans, n, k,                                     \
                              (V) alpha, (const V*) A, lda,                                 \
                              (V) beta, (V*) C, ldc);                                       \
}

#define CUSP_CBLAS_SYR2K(T,V,name)                                                          \
  void syr2k( enum CBLAS_ORDER order,                                                       \
                        enum CBLAS_UPLO uplo, enum CBLAS_TRANSPOSE trans,                   \
                        int n, int k, T alpha, const T* A, int lda,                         \
                        const T* B, int ldb, T beta, T* C, int ldc)                         \
{                                                                                           \
    return cblas_##name##syr2k(order, uplo, trans, n, k,                                    \
                               (V) alpha, (const V*) A, lda,                                \
                               (const V*) B, ldb, (V) beta, (V*) C, ldc);                   \
}

#define CUSP_CBLAS_TRMM(T,V,name)                                                           \
  void trmm( enum CBLAS_ORDER order,                                                        \
                       enum CBLAS_SIDE side, enum CBLAS_UPLO uplo,                          \
                       enum CBLAS_TRANSPOSE trans, enum CBLAS_DIAG diag,                    \
                       int m, int n, T alpha, const T* A, int lda,                          \
                       T* B, int ldb)                                                       \
{                                                                                           \
    return cblas_##name##trmm(order, side, uplo, trans, diag, m, n,                         \
                              (V) alpha, (const V*) A, lda, (V*) B, ldb);                   \
}

#define CUSP_CBLAS_TRSM(T,V,name)                                                           \
  void trsm( enum CBLAS_ORDER order,                                                        \
                       enum CBLAS_SIDE side, enum CBLAS_UPLO uplo,                          \
                       enum CBLAS_TRANSPOSE trans, enum CBLAS_DIAG diag,                    \
                       int m, int n, T alpha, const T* A, int lda,                          \
                       T* B, int ldb)                                                       \
{                                                                                           \
    return cblas_##name##trsm(order, side, uplo, trans, diag, m, n,                         \
                              (V) alpha, (const V*) A, lda, (V*) B, ldb);                   \
}

namespace cusp
{
namespace blas
{
namespace cblas
{
namespace detail
{

// LEVEL 1
CUSP_CBLAS_EXPAND_DEFS_1(CUSP_CBLAS_AMAX);
CUSP_CBLAS_EXPAND_DEFS_2(CUSP_CBLAS_ASUM);
CUSP_CBLAS_EXPAND_DEFS_1(CUSP_CBLAS_AXPY);
CUSP_CBLAS_EXPAND_DEFS_1(CUSP_CBLAS_COPY);
CUSP_CBLAS_EXPAND_REAL_DEFS(CUSP_CBLAS_DOT);
// CUSP_CBLAS_EXPAND_COMPLEX_DEFS_1(CUSP_CBLAS_DOTC);
// CUSP_CBLAS_EXPAND_COMPLEX_DEFS_1(CUSP_CBLAS_DOTU);
CUSP_CBLAS_EXPAND_DEFS_1(CUSP_CBLAS_NRM2);
CUSP_CBLAS_EXPAND_DEFS_1(CUSP_CBLAS_SCAL);
CUSP_CBLAS_EXPAND_DEFS_1(CUSP_CBLAS_SWAP);

// LEVEL 2
CUSP_CBLAS_EXPAND_DEFS_1(CUSP_CBLAS_GEMV);
CUSP_CBLAS_EXPAND_REAL_DEFS(CUSP_CBLAS_GER);
CUSP_CBLAS_EXPAND_DEFS_1(CUSP_CBLAS_SYMV);
CUSP_CBLAS_EXPAND_DEFS_1(CUSP_CBLAS_SYR);
CUSP_CBLAS_EXPAND_DEFS_1(CUSP_CBLAS_TRMV);
CUSP_CBLAS_EXPAND_DEFS_1(CUSP_CBLAS_TRSV);

// LEVEL 3
CUSP_CBLAS_EXPAND_DEFS_1(CUSP_CBLAS_GEMM);
CUSP_CBLAS_EXPAND_DEFS_1(CUSP_CBLAS_SYMM);
CUSP_CBLAS_EXPAND_DEFS_1(CUSP_CBLAS_SYRK);
CUSP_CBLAS_EXPAND_DEFS_1(CUSP_CBLAS_SYR2K);
CUSP_CBLAS_EXPAND_DEFS_1(CUSP_CBLAS_TRMM);
CUSP_CBLAS_EXPAND_DEFS_1(CUSP_CBLAS_TRSM);

} // end namespace detail
} // end namespace cblas
} // end namespace blas
} // end namespace cusp

