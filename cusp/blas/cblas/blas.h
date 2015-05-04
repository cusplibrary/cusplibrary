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
#include <cusp/blas/cblas/execution_policy.h>

namespace cusp
{
namespace blas
{
namespace cblas
{

template <typename DerivedPolicy,
          typename Array>
int amax(cblas::execution_policy<DerivedPolicy>& policy,
         const Array& x)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    return detail::amax(n, x_p, 1);
}

template <typename DerivedPolicy,
          typename Array>
typename Array::value_type
asum(cblas::execution_policy<DerivedPolicy>& policy,
     const Array& x)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    return detail::asum(n, x_p, 1);
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2,
          typename ScalarType>
void axpy(cblas::execution_policy<DerivedPolicy>& policy,
          const Array1& x,
                Array2& y,
                ScalarType alpha)
{
    typedef typename Array1::value_type ValueType;

    int n = y.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    detail::axpy(n, ValueType(alpha), x_p, 1, y_p, 1);
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
void copy(cblas::execution_policy<DerivedPolicy>& policy,
          const Array1& x,
                Array2& y)
{
    typedef typename Array1::value_type ValueType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    detail::copy(n, x_p, 1, y_p, 1);
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
typename Array1::value_type
dot(cblas::execution_policy<DerivedPolicy>& policy,
    const Array1& x,
    const Array2& y)
{
    typedef typename Array2::value_type ValueType;

    int n = y.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    const ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    return detail::dot(n, x_p, 1, y_p, 1);
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
typename Array1::value_type
dotc(cblas::execution_policy<DerivedPolicy>& policy,
     const Array1& x,
     const Array2& y)
{
    typedef typename Array2::value_type ValueType;

    int n = y.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    const ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    ValueType result;

    // cblas::detail::dotc(n, x_p, 1, y_p, 1, &result);

    return result;
}

template <typename DerivedPolicy,
          typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm2(cblas::execution_policy<DerivedPolicy>& policy,
     const Array& x)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    return detail::nrm2(n, x_p, 1);
}

template <typename DerivedPolicy,
          typename Array,
          typename ScalarType>
void scal(cblas::execution_policy<DerivedPolicy>& policy,
          Array& x,
          ScalarType alpha)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    detail::scal(n, ValueType(alpha), x_p, 1);
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
void swap(cblas::execution_policy<DerivedPolicy>& policy,
          Array1& x,
          Array2& y)
{
    typedef typename Array1::value_type ValueType;

    int n = x.size();

    ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    detail::swap(n, x_p, 1, y_p, 1);
}

template<typename DerivedPolicy,
         typename Array2d1,
         typename Array1d1,
         typename Array1d2>
void gemv(cblas::execution_policy<DerivedPolicy>& policy,
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

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType * x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType * y_p = thrust::raw_pointer_cast(&y[0]);

    detail::gemv(order, trans, m, n, alpha,
                 A_p, m, x_p, 1, beta, y_p, 1);
}

template<typename DerivedPolicy,
         typename Array1d1,
         typename Array1d2,
         typename Array2d1,
         typename ScalarType>
void ger(cblas::execution_policy<DerivedPolicy>& policy,
         const Array1d1& x,
         const Array1d2& y,
               Array2d1& A,
               ScalarType alpha)
{
    typedef typename Array2d1::value_type ValueType;

    enum CBLAS_ORDER order = CblasColMajor;

    int m = A.num_rows;
    int n = A.num_cols;
    int lda = A.pitch;

    const ValueType * x_p = thrust::raw_pointer_cast(&x[0]);
    const ValueType * y_p = thrust::raw_pointer_cast(&y[0]);
    ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));

    detail::ger(order, m, n, ValueType(alpha),
                x_p, 1, y_p, 1, A_p, lda);
}

template<typename DerivedPolicy,
         typename Array2d1,
         typename Array1d1,
         typename Array1d2>
void symv(cblas::execution_policy<DerivedPolicy>& policy,
          const Array2d1& A,
          const Array1d1& x,
                Array1d2& y)
{
    typedef typename Array2d1::value_type ValueType;

    enum CBLAS_ORDER order = CblasColMajor;
    enum CBLAS_UPLO  uplo  = CblasUpper;

    int n = A.num_rows;
    int lda = A.pitch;

    ValueType alpha = 1.0;
    ValueType beta = 0.0;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType * x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType * y_p = thrust::raw_pointer_cast(&y[0]);

    detail::symv(order, uplo, n, alpha,
                 A_p, lda, x_p, 1, beta, y_p, 1);
}

template<typename DerivedPolicy,
         typename Array1d,
         typename Array2d,
         typename ScalarType>
void syr(cblas::execution_policy<DerivedPolicy>& policy,
         const Array1d& x,
               Array2d& A,
               ScalarType alpha)
{
    typedef typename Array2d::value_type ValueType;

    enum CBLAS_ORDER order = CblasColMajor;
    enum CBLAS_UPLO  uplo  = CblasUpper;

    int n = A.num_rows;
    int lda = A.pitch;

    const ValueType * x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));

    detail::syr(order, uplo, n, ValueType(alpha),
                x_p, 1, A_p, lda);
}

template<typename DerivedPolicy,
         typename Array2d,
         typename Array1d>
void trmv(cblas::execution_policy<DerivedPolicy>& policy,
          const Array2d& A,
                Array1d& x)
{
    typedef typename Array2d::value_type ValueType;

    enum CBLAS_ORDER     order = CblasColMajor;
    enum CBLAS_UPLO      uplo  = CblasUpper;
    enum CBLAS_TRANSPOSE trans = CblasNoTrans;
    enum CBLAS_DIAG      diag  = CblasNonUnit;

    int n = A.num_rows;
    int lda = A.pitch;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    ValueType * x_p = thrust::raw_pointer_cast(&x[0]);

    detail::trmv(order, uplo, trans, diag, n,
                 A_p, lda, x_p, 1);
}

template<typename DerivedPolicy,
         typename Array2d,
         typename Array1d>
void trsv(cblas::execution_policy<DerivedPolicy>& policy,
          const Array2d& A,
                Array1d& x)
{
    typedef typename Array2d::value_type ValueType;

    enum CBLAS_ORDER     order = CblasColMajor;
    enum CBLAS_UPLO      uplo  = CblasUpper;
    enum CBLAS_TRANSPOSE trans = CblasNoTrans;
    enum CBLAS_DIAG      diag  = CblasNonUnit;

    int n = A.num_rows;
    int lda = A.pitch;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    ValueType * x_p = thrust::raw_pointer_cast(&x[0]);

    detail::trsv(order, uplo, trans, diag, n,
                 A_p, lda, x_p, 1);
}

template<typename DerivedPolicy,
         typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void gemm(cblas::execution_policy<DerivedPolicy>& policy,
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

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    const ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));
    ValueType * C_p = thrust::raw_pointer_cast(&C(0,0));

    detail::gemm(order, transa, transb,
                 m, n, k, alpha, A_p, m,
                 B_p, k, beta, C_p, m);
}

template<typename DerivedPolicy,
         typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void symm(cblas::execution_policy<DerivedPolicy>& policy,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C)
{
    typedef typename Array2d1::value_type ValueType;

    enum CBLAS_ORDER order = CblasColMajor;
    enum CBLAS_SIDE  side  = CblasLeft;
    enum CBLAS_UPLO  uplo  = CblasUpper;

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

    detail::symm(order, side, uplo,
                 m, n, alpha, A_p, lda,
                 B_p, ldb, beta, C_p, ldc);
}

template<typename DerivedPolicy,
         typename Array2d1,
         typename Array2d2>
void syrk(cblas::execution_policy<DerivedPolicy>& policy,
          const Array2d1& A,
                Array2d2& B)
{
    typedef typename Array2d1::value_type ValueType;

    enum CBLAS_ORDER order = CblasColMajor;
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

    detail::syrk(order, uplo, trans,
                 n, k, alpha, A_p, lda,
                 beta, B_p, ldb);
}

template<typename DerivedPolicy,
         typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void syr2k(cblas::execution_policy<DerivedPolicy>& policy,
           const Array2d1& A,
           const Array2d2& B,
                 Array2d3& C)
{
    typedef typename Array2d1::value_type ValueType;

    enum CBLAS_ORDER order = CblasColMajor;
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

    detail::syr2k(order, uplo, trans,
                  n, k, alpha, A_p, lda,
                  beta, B_p, ldb, C_p, ldc);
}

template<typename DerivedPolicy,
         typename Array2d1,
         typename Array2d2>
void trmm(cblas::execution_policy<DerivedPolicy>& policy,
          const Array2d1& A,
                Array2d2& B)
{
    typedef typename Array2d1::value_type ValueType;

    enum CBLAS_ORDER order = CblasColMajor;
    enum CBLAS_SIDE  side  = CblasLeft;
    enum CBLAS_UPLO  uplo  = CblasUpper;
    enum CBLAS_TRANSPOSE trans = CblasNoTrans;
    enum CBLAS_DIAG  diag  = CblasNonUnit;

    int m = A.num_rows;
    int n = A.num_cols;
    int lda = A.pitch;
    int ldb = B.pitch;

    ValueType alpha = 1.0;
    ValueType beta = 0.0;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));

    detail::trmm(order, side, uplo, trans, diag,
                 m, n, alpha, A_p, lda, B_p, ldb);
}

template<typename DerivedPolicy,
         typename Array2d1,
         typename Array2d2>
void trsm(cblas::execution_policy<DerivedPolicy>& policy,
          const Array2d1& A,
                Array2d2& B)
{
    typedef typename Array2d1::value_type ValueType;

    enum CBLAS_ORDER order = CblasColMajor;
    enum CBLAS_SIDE  side  = CblasLeft;
    enum CBLAS_UPLO  uplo  = CblasUpper;
    enum CBLAS_TRANSPOSE trans = CblasNoTrans;
    enum CBLAS_DIAG  diag  = CblasNonUnit;

    int m = A.num_rows;
    int n = A.num_cols;
    int lda = A.pitch;
    int ldb = B.pitch;

    ValueType alpha = 1.0;
    ValueType beta = 0.0;

    const ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));

    detail::trsm(order, side, uplo, trans, diag,
                 m, n, alpha, A_p, lda, B_p, ldb);
}

template <typename DerivedPolicy,
          typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm1(cblas::execution_policy<DerivedPolicy>& policy,
     const Array& x)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    return detail::asum(n, x_p, 1);
}

template <typename DerivedPolicy,
          typename Array>
typename Array::value_type
nrmmax(cblas::execution_policy<DerivedPolicy>& policy,
       const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename cusp::detail::norm_type<ValueType>::type ResultType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    int index = cblas::detail::amax(n, x_p, 1);

    return x[index];
}

} // end namespace cblas

using cblas::amax;
using cblas::asum;
using cblas::axpy;
using cblas::copy;
using cblas::dot;
using cblas::dotc;
using cblas::nrm2;
using cblas::scal;
using cblas::swap;

using cblas::gemv;
using cblas::ger;
using cblas::symv;
using cblas::syr;
using cblas::trmv;
using cblas::trsv;

using cblas::gemm;
using cblas::symm;
using cblas::syrk;
using cblas::syr2k;
using cblas::trmm;
using cblas::trsm;

using cblas::nrm1;
using cblas::nrmmax;

} // end namespace blas
} // end namespace cusp

