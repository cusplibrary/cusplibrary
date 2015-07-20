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

#include <cusp/blas/cblas/defs.h>
#include <cusp/blas/cblas/execution_policy.h>

#include <cusp/blas/cblas/complex_stubs.h>
#include <cusp/blas/cblas/stubs.h>

namespace cusp
{
namespace blas
{
namespace cblas
{

template <typename Array>
int amax(cblas::execution_policy& exec,
         const Array& x)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    return cblas::detail::amax(n, x_p, 1);
}

template <typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
asum(cblas::execution_policy& exec,
     const Array& x)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    return cblas::detail::asum(n, x_p, 1);
}

template <typename Array1,
          typename Array2,
          typename ScalarType>
void axpy(cblas::execution_policy& exec,
          const Array1& x,
                Array2& y,
          const ScalarType alpha)
{
    typedef typename Array1::value_type ValueType;

    int n = y.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    cblas::detail::axpy(n, alpha, x_p, 1, y_p, 1);
}

template <typename Array1,
          typename Array2>
void copy(cblas::execution_policy& exec,
          const Array1& x,
                Array2& y)
{
    typedef typename Array1::value_type ValueType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    cblas::detail::copy(n, x_p, 1, y_p, 1);
}

template <typename Array1,
          typename Array2>
typename Array1::value_type
dot(cblas::execution_policy& exec,
    const Array1& x,
    const Array2& y)
{
    typedef typename Array2::value_type ValueType;

    int n = y.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    const ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    return cblas::detail::dot(n, x_p, 1, y_p, 1);
}

// template <typename Array1,
//           typename Array2>
// typename Array1::value_type
// dotc(cblas::execution_policy& exec,
//      const Array1& x,
//      const Array2& y)
// {
//     typedef typename Array2::value_type ValueType;
//
//     int n = y.size();
//
//     const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
//     const ValueType* y_p = thrust::raw_pointer_cast(&y[0]);
//
//     return cblas::detail::dotc(n, x_p, 1, y_p, 1);
// }

template <typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm2(cblas::execution_policy& exec,
     const Array& x)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    return cblas::detail::nrm2(n, x_p, 1);
}

template <typename Array,
          typename ScalarType>
void scal(cblas::execution_policy& exec,
          Array& x,
          const ScalarType alpha)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    cblas::detail::scal(n, alpha, x_p, 1);
}

template <typename Array1,
          typename Array2>
void swap(cblas::execution_policy& exec,
          Array1& x,
          Array2& y)
{
    typedef typename Array1::value_type ValueType;

    int n = x.size();

    ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    cblas::detail::swap(n, x_p, 1, y_p, 1);
}

template<typename Array2d1,
         typename Array1d1,
         typename Array1d2>
void gemv(cblas::execution_policy& exec,
          const Array2d1& A,
          const Array1d1& x,
                Array1d2& y)
{
    typedef typename Array2d1::value_type ValueType;

    enum CBLAS_ORDER order = cblas::Orientation<typename Array2d1::orientation>::type;
    enum CBLAS_TRANSPOSE trans = CblasNoTrans;

    int m = A.num_rows;
    int n = A.num_cols;
    int lda = A.pitch;

    ValueType alpha = 1.0;
    ValueType beta = 0.0;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType * x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType * y_p = thrust::raw_pointer_cast(&y[0]);

    cblas::detail::gemv(order, trans, m, n, alpha,
                        A_p, lda, x_p, 1, beta, y_p, 1);
}

template<typename Array1d1,
         typename Array1d2,
         typename Array2d1>
void ger(cblas::execution_policy& exec,
         const Array1d1& x,
         const Array1d2& y,
               Array2d1& A)
{
    typedef typename Array2d1::value_type ValueType;

    enum CBLAS_ORDER order = cblas::Orientation<typename Array2d1::orientation>::type;

    int m = A.num_rows;
    int n = A.num_cols;
    int lda = A.pitch;

    ValueType alpha = 1.0;

    const ValueType * x_p = thrust::raw_pointer_cast(&x[0]);
    const ValueType * y_p = thrust::raw_pointer_cast(&y[0]);
    ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));

    cblas::detail::ger(order, m, n, alpha,
                       x_p, 1, y_p, 1, A_p, lda);
}

template<typename Array2d1,
         typename Array1d1,
         typename Array1d2>
void symv(cblas::execution_policy& exec,
          const Array2d1& A,
          const Array1d1& x,
                Array1d2& y)
{
    typedef typename Array2d1::value_type ValueType;

    enum CBLAS_ORDER order = cblas::Orientation<typename Array2d1::orientation>::type;
    enum CBLAS_UPLO  uplo  = CblasUpper;

    int n = A.num_rows;
    int lda = A.pitch;

    ValueType alpha = 1.0;
    ValueType beta = 0.0;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType * x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType * y_p = thrust::raw_pointer_cast(&y[0]);

    cblas::detail::symv(order, uplo, n, alpha,
                        A_p, lda, x_p, 1, beta, y_p, 1);
}

template<typename Array1d,
         typename Array2d>
void syr(cblas::execution_policy& exec,
         const Array1d& x,
               Array2d& A)
{
    typedef typename Array2d::value_type ValueType;

    enum CBLAS_ORDER order = cblas::Orientation<typename Array2d::orientation>::type;
    enum CBLAS_UPLO  uplo  = CblasUpper;

    int n = A.num_rows;
    int lda = A.pitch;

    const ValueType * x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));

    ValueType alpha = 1.0;

    cblas::detail::syr(order, uplo, n, alpha,
                       x_p, 1, A_p, lda);
}

template<typename Array2d,
         typename Array1d>
void trmv(cblas::execution_policy& exec,
          const Array2d& A,
                Array1d& x)
{
    typedef typename Array2d::value_type ValueType;

    enum CBLAS_ORDER order = cblas::Orientation<typename Array2d::orientation>::type;
    enum CBLAS_UPLO      uplo  = CblasUpper;
    enum CBLAS_TRANSPOSE trans = CblasNoTrans;
    enum CBLAS_DIAG      diag  = CblasNonUnit;

    int n = A.num_rows;
    int lda = A.pitch;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    ValueType * x_p = thrust::raw_pointer_cast(&x[0]);

    cblas::detail::trmv(order, uplo, trans, diag, n,
                        A_p, lda, x_p, 1);
}

template<typename Array2d,
         typename Array1d>
void trsv(cblas::execution_policy& exec,
          const Array2d& A,
                Array1d& x)
{
    typedef typename Array2d::value_type ValueType;

    enum CBLAS_ORDER order = cblas::Orientation<typename Array2d::orientation>::type;
    enum CBLAS_UPLO      uplo  = CblasUpper;
    enum CBLAS_TRANSPOSE trans = CblasNoTrans;
    enum CBLAS_DIAG      diag  = CblasNonUnit;

    int n = A.num_rows;
    int lda = A.pitch;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    ValueType * x_p = thrust::raw_pointer_cast(&x[0]);

    cblas::detail::trsv(order, uplo, trans, diag, n,
                        A_p, lda, x_p, 1);
}

template<typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void gemm(cblas::execution_policy& exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C)
{
    typedef typename Array2d1::value_type ValueType;

    enum CBLAS_ORDER order = cblas::Orientation<typename Array2d1::orientation>::type;
    enum CBLAS_TRANSPOSE transa = CblasNoTrans;
    enum CBLAS_TRANSPOSE transb = CblasNoTrans;

    int m = C.num_rows;
    int n = C.num_cols;
    int k = B.num_rows;
    int lda = A.pitch;
    int ldb = B.pitch;
    int ldc = C.pitch;

    ValueType alpha = 1.0;
    ValueType beta = 0.0;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));
    ValueType * C_p = thrust::raw_pointer_cast(&C(0,0));

    cblas::detail::gemm(order, transa, transb,
                        m, n, k, alpha, A_p, lda,
                        B_p, ldb, beta, C_p, ldc);
}

template<typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void symm(cblas::execution_policy& exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C)
{
    typedef typename Array2d1::value_type ValueType;

    enum CBLAS_ORDER order = cblas::Orientation<typename Array2d1::orientation>::type;
    enum CBLAS_SIDE  side  = CblasLeft;
    enum CBLAS_UPLO  uplo  = CblasUpper;

    int m = C.num_rows;
    int n = C.num_cols;
    int lda = A.pitch;
    int ldb = B.pitch;
    int ldc = C.pitch;

    ValueType alpha = 1.0;
    ValueType beta = 0.0;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));
    ValueType * C_p = thrust::raw_pointer_cast(&C(0,0));

    cblas::detail::symm(order, side, uplo,
                        m, n, alpha, A_p, lda,
                        B_p, ldb, beta, C_p, ldc);
}

template<typename Array2d1,
         typename Array2d2>
void syrk(cblas::execution_policy& exec,
          const Array2d1& A,
                Array2d2& B)
{
    typedef typename Array2d1::value_type ValueType;

    enum CBLAS_ORDER order = cblas::Orientation<typename Array2d1::orientation>::type;
    enum CBLAS_UPLO  uplo  = CblasUpper;
    enum CBLAS_TRANSPOSE trans = CblasNoTrans;

    int n = A.num_rows;
    int k = B.num_rows;
    int lda = A.pitch;
    int ldb = B.pitch;

    ValueType alpha = 1.0;
    ValueType beta = 0.0;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));

    cblas::detail::syrk(order, uplo, trans,
                        n, k, alpha, A_p, lda,
                        beta, B_p, ldb);
}

template<typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void syr2k(cblas::execution_policy& exec,
           const Array2d1& A,
           const Array2d2& B,
                 Array2d3& C)
{
    typedef typename Array2d1::value_type ValueType;

    enum CBLAS_ORDER order = cblas::Orientation<typename Array2d1::orientation>::type;
    enum CBLAS_UPLO  uplo  = CblasUpper;
    enum CBLAS_TRANSPOSE trans = CblasNoTrans;

    int n = A.num_rows;
    int k = B.num_rows;
    int lda = A.pitch;
    int ldb = B.pitch;
    int ldc = C.pitch;

    ValueType alpha = 1.0;
    ValueType beta = 0.0;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));
    ValueType * C_p = thrust::raw_pointer_cast(&C(0,0));

    cblas::detail::syr2k(order, uplo, trans,
                         n, k, alpha, A_p, lda,
                         B_p, ldb, beta, C_p, ldc);
}

template<typename Array2d1,
         typename Array2d2>
void trmm(cblas::execution_policy& exec,
          const Array2d1& A,
                Array2d2& B)
{
    typedef typename Array2d1::value_type ValueType;

    enum CBLAS_ORDER order     = cblas::Orientation<typename Array2d1::orientation>::type;
    enum CBLAS_SIDE  side      = CblasLeft;
    enum CBLAS_UPLO  uplo      = CblasUpper;
    enum CBLAS_TRANSPOSE trans = CblasNoTrans;
    enum CBLAS_DIAG  diag      = CblasNonUnit;

    int m   = B.num_rows;
    int n   = B.num_cols;
    int lda = A.pitch;
    int ldb = B.pitch;

    ValueType alpha = 1.0;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));

    cblas::detail::trmm(order, side, uplo, trans, diag,
                        m, n, alpha, A_p, lda, B_p, ldb);
}

template<typename Array2d1,
         typename Array2d2>
void trsm(cblas::execution_policy& exec,
          const Array2d1& A,
                Array2d2& B)
{
    typedef typename Array2d1::value_type ValueType;

    enum CBLAS_ORDER order     = cblas::Orientation<typename Array2d1::orientation>::type;
    enum CBLAS_SIDE  side      = CblasLeft;
    enum CBLAS_UPLO  uplo      = CblasUpper;
    enum CBLAS_TRANSPOSE trans = CblasNoTrans;
    enum CBLAS_DIAG  diag      = CblasNonUnit;

    int m   = B.num_rows;
    int n   = B.num_cols;
    int lda = A.pitch;
    int ldb = B.pitch;

    ValueType alpha = 1.0;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));

    cblas::detail::trsm(order, side, uplo, trans, diag,
                        m, n, alpha, A_p, lda, B_p, ldb);
}

template <typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm1(cblas::execution_policy& exec,
     const Array& x)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    return cblas::detail::asum(n, x_p, 1);
}

template <typename Array>
typename cusp::detail::norm_type<typename ArrayType::value_type>::type
nrmmax(cblas::execution_policy& exec,
       const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename cusp::detail::norm_type<ValueType>::type ResultType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    int index = cblas::detail::amax(n, x_p, 1);

    return cusp::norm(x[index]);
}

} // end namespace cblas
} // end namespace blas
} // end namespace cusp

