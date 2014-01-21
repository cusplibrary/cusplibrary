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

#pragma once

#include <cusp/array1d.h>
#include <cusp/exception.h>
#include <cusp/blas/cublas/stubs.h>

#include <cublas_v2.h>

namespace cusp
{
namespace blas
{
namespace cublas
{

class cublasLibrary
{
public:

    cublasHandle_t handle;

    cublasLibrary(void)
    {
        if(cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
            throw cusp::runtime_exception("cublasCreate failed!");
    }

    ~cublasLibrary(void)
    {
        // if(cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS)
        //     throw cusp::runtime_exception("cublasDestroy failed!");
    }
};

static cublasLibrary __cublas;
} // end namespace cublas

template <typename Array1,
         typename Array2,
         typename ScalarType>
void axpy(const cublas::detail::blas_policy<typename Array2::memory_space>& policy,
          const Array1& x,
          Array2& y,
          ScalarType alpha)
{
    typedef typename Array2::value_type ValueType;

    int n = y.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    if(cublas::detail::axpy(cublas::__cublas.handle, n, &alpha, x_p, 1, y_p, 1) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS axpy failed!");
    }
}

template <typename Array1,
         typename Array2,
         typename Array3,
         typename ScalarType1,
         typename ScalarType2>
void axpby(const cublas::detail::blas_policy<typename Array3::memory_space>& policy,
           const Array1& x,
           const Array2& y,
           Array3& z,
           ScalarType1 alpha,
           ScalarType2 beta)
{
    typedef typename Array3::value_type ValueType;

    int n = z.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    const ValueType* y_p = thrust::raw_pointer_cast(&y[0]);
    ValueType* z_p = thrust::raw_pointer_cast(&z[0]);

    if(cublas::detail::axpy(cublas::__cublas.handle, n, &alpha, x_p, 1, z_p, 1) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS axpby failed!");
    }
    if(cublas::detail::axpy(cublas::__cublas.handle, n, &beta, y_p, 1, z_p, 1) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS axpby failed!");
    }
}

template <typename Array1,
         typename Array2,
         typename Array3,
         typename Array4,
         typename ScalarType1,
         typename ScalarType2,
         typename ScalarType3>
void axpbypcz(const cublas::detail::blas_policy<typename Array4::memory_space>& policy,
              const Array1& x,
              const Array2& y,
              const Array3& z,
              Array4& output,
              ScalarType1 alpha,
              ScalarType2 beta,
              ScalarType3 gamma)
{
    typedef typename Array3::value_type ValueType;

    int n = output.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    const ValueType* y_p = thrust::raw_pointer_cast(&y[0]);
    const ValueType* z_p = thrust::raw_pointer_cast(&z[0]);
    ValueType* output_p = thrust::raw_pointer_cast(&output[0]);

    if(cublas::detail::axpy(cublas::__cublas.handle, n, &alpha, x_p, 1, output_p, 1) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS axpbypcz failed!");
    }
    if(cublas::detail::axpy(cublas::__cublas.handle, n, &beta, y_p, 1, output_p, 1) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS axpbypcz failed!");
    }
    if(cublas::detail::axpy(cublas::__cublas.handle, n, &gamma, z_p, 1, output_p, 1) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS axpbypcz failed!");
    }
}

template <typename Array1,
         typename Array2,
         typename Array3>
void xmy(const cublas::detail::blas_policy<typename Array3::memory_space>& policy,
         const Array1& x,
         const Array2& y,
         Array3& output)
{
    typedef typename Array3::value_type ValueType;

    int n = x.size();

    output = y;

    const ValueType *x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType *output_p = thrust::raw_pointer_cast(&output[0]);

    if(cublas::detail::xmy(cublas::__cublas.handle, n, x_p, output_p) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS xmy failed!");
    }
}

template <typename Array1,
         typename Array2>
typename Array1::value_type
dot(const cublas::detail::blas_policy<typename Array1::memory_space>& policy,
    const Array1& x,
    const Array2& y)
{
    typedef typename Array2::value_type ValueType;

    int n = y.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    const ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    ValueType result;

    if(cublas::detail::dot(cublas::__cublas.handle, n, x_p, 1, y_p, 1, &result) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS dot failed!");
    }

    return result;
}

template <typename Array1,
         typename Array2>
typename Array1::value_type
dotc(const cublas::detail::blas_policy<typename Array1::memory_space>& policy,
    const Array1& x,
    const Array2& y)
{
    typedef typename Array2::value_type ValueType;

    return ValueType(0);

    // int n = y.size();

    // const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    // const ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    // ValueType result;

    // if(cublas::detail::dotc(cublas::__cublas.handle, n, x_p, 1, y_p, 1, &result) != CUBLAS_STATUS_SUCCESS)
    // {
    //     throw cusp::runtime_exception("CUBLAS dotc failed!");
    // }

    // return result;
}

template <typename Array>
typename norm_type<typename Array::value_type>::type
nrm1(const cublas::detail::blas_policy<typename Array::memory_space>& policy,
     const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename norm_type<ValueType>::type ResultType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    ResultType result;

    if(cublas::detail::asum(cublas::__cublas.handle, n, x_p, 1, &result) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS asum failed!");
    }

    return result;
}

template <typename Array>
typename norm_type<typename Array::value_type>::type
nrm2(const cublas::detail::blas_policy<typename Array::memory_space>& policy,
     const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename norm_type<ValueType>::type ResultType;

    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    ResultType result;

    if(cublas::detail::nrm2(cublas::__cublas.handle, n, x_p, 1, &result) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS nrm2 failed!");
    }

    return result;
}

template <typename Array>
typename Array::value_type
nrmmax(const cublas::detail::blas_policy<typename Array::memory_space>& policy,
       const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename norm_type<ValueType>::type ResultType;

    int index;
    int n = x.size();

    const ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    if(cublas::detail::amax(cublas::__cublas.handle, n, x_p, 1, &index) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS amax failed!");
    }

    return x[index];
}

template <typename Array, typename ScalarType>
void scal(const cublas::detail::blas_policy<typename Array::memory_space>& policy,
          Array& x,
          ScalarType alpha)
{
    typedef typename Array::value_type ValueType;

    int n = x.size();

    ValueType* x_p = thrust::raw_pointer_cast(&x[0]);

    if(cublas::detail::scal(cublas::__cublas.handle, n, &alpha, x_p, 1) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS scal failed!");
    }
}

template<typename Array2d1, typename Array1d1, typename Array1d2>
void gemv(const cublas::detail::blas_policy<typename Array1d2::memory_space>& policy,
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

    ValueType *A_p = thrust::raw_pointer_cast(&A[0]);
    ValueType *x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType *y_p = thrust::raw_pointer_cast(&y[0]);

    cublasStatus_t result;

    result = cublas::detail::gemv(cublas::__cublas.handle, trans, m, n, &alpha,
                                  A_p, m, x_p, 1, &beta, y_p, 1);

    if(result != CUBLAS_STATUS_SUCCESS)
        throw cusp::runtime_exception("CUBLAS gemv failed!");
}

template<typename Array2d1, typename Array2d2, typename Array2d3>
void gemm(const cublas::detail::blas_policy<typename Array2d3::memory_space>& policy,
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

    ValueType * A_p = thrust::raw_pointer_cast(&A(0,0));
    ValueType * B_p = thrust::raw_pointer_cast(&B(0,0));
    ValueType * C_p = thrust::raw_pointer_cast(&C(0,0));

    cublasStatus_t result;

    result = cublas::detail::gemm(cublas::__cublas.handle, transa, transb,
                                  m, n, k, &alpha, A_p, m,
                                  B_p, k, &beta, C_p, m);

    if(result != CUBLAS_STATUS_SUCCESS)
        throw cusp::runtime_exception("CUBLAS gemm failed!");
}

} // end namespace blas
} // end namespace cusp

