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

cublasLibrary __cublas;

template <typename Array1,
         typename Array2>
void axpy(const Array1& x,
          Array2& y,
          typename Array1::value_type alpha,
          cusp::device_memory)
{
    typedef typename Array1::value_type ValueType;

    CUSP_PROFILE_SCOPED();

    int m = x.size();
    int n = y.size();

    ValueType* x_p = thrust::raw_pointer_cast(&x[0]);
    ValueType* y_p = thrust::raw_pointer_cast(&y[0]);

    if(detail::axpy(__cublas.handle, n, &alpha, x_p, 1, y_p, 1) != CUBLAS_STATUS_SUCCESS)
    {
        throw cusp::runtime_exception("CUBLAS axpy failed!");
    }
}

template <typename Array1,
         typename Array2,
         typename ScalarType>
void axpy(const Array1& x,
          const Array2& y,
          ScalarType alpha,
          cusp::device_memory)
{
    CUSP_PROFILE_SCOPED();
    throw cusp::not_implemented_exception("CUBLAS axpy not implemented");
}

template <typename Array1,
         typename Array2,
         typename Array3,
         typename ScalarType1,
         typename ScalarType2>
void axpby(const Array1& x,
           const Array2& y,
           Array3& z,
           ScalarType1 alpha,
           ScalarType2 beta,
           cusp::device_memory)
{
    CUSP_PROFILE_SCOPED();
    throw cusp::not_implemented_exception("CUBLAS axpby not implemented");
}

template <typename Array1,
         typename Array2,
         typename Array3,
         typename ScalarType1,
         typename ScalarType2>
void axpby(const Array1& x,
           const Array2& y,
           const Array3& z,
           ScalarType1 alpha,
           ScalarType2 beta,
           cusp::device_memory)
{
    CUSP_PROFILE_SCOPED();
    throw cusp::not_implemented_exception("CUBLAS axpby not implemented");
}

template <typename Array1,
         typename Array2,
         typename Array3,
         typename Array4,
         typename ScalarType1,
         typename ScalarType2,
         typename ScalarType3>
void axpbypcz(const Array1& x,
              const Array2& y,
              const Array3& z,
              Array4& output,
              ScalarType1 alpha,
              ScalarType2 beta,
              ScalarType3 gamma,
              cusp::device_memory)
{
    CUSP_PROFILE_SCOPED();
    throw cusp::not_implemented_exception("CUBLAS axpbypcz not implemented");
}

template <typename Array1,
         typename Array2,
         typename Array3,
         typename Array4,
         typename ScalarType1,
         typename ScalarType2,
         typename ScalarType3>
void axpbypcz(const Array1& x,
              const Array2& y,
              const Array3& z,
              const Array4& output,
              ScalarType1 alpha,
              ScalarType2 beta,
              ScalarType3 gamma,
              cusp::device_memory)
{
    CUSP_PROFILE_SCOPED();
    throw cusp::not_implemented_exception("CUBLAS axpbypcz not implemented");
}

template <typename Array1,
         typename Array2,
         typename Array3>
void xmy(const Array1& x,
         const Array2& y,
         Array3& output,
         cusp::device_memory)
{
    CUSP_PROFILE_SCOPED();
    throw cusp::not_implemented_exception("CUBLAS xmy not implemented");
}

template <typename Array1,
         typename Array2,
         typename Array3>
void xmy(const Array1& x,
         const Array2& y,
         const Array3& output,
         cusp::device_memory)
{
    CUSP_PROFILE_SCOPED();
    throw cusp::not_implemented_exception("CUBLAS xmy not implemented");
}

template <typename Array1,
         typename Array2>
void copy(const Array1& x,
          Array2& y,
          cusp::device_memory)
{
    CUSP_PROFILE_SCOPED();
    throw cusp::not_implemented_exception("CUBLAS copy not implemented");
}

template <typename Array1,
         typename Array2>
void copy(const Array1& x,
          const Array2& y,
          cusp::device_memory)
{
    CUSP_PROFILE_SCOPED();
    throw cusp::not_implemented_exception("CUBLAS copy not implemented");
}

template <typename Array1,
         typename Array2>
typename Array1::value_type
dot(const Array1& x,
    const Array2& y,
    cusp::device_memory)
{
    CUSP_PROFILE_SCOPED();
    throw cusp::not_implemented_exception("CUBLAS dot not implemented");

    return 0;
}

template <typename Array>
void fill(Array& x,
          typename Array::value_type alpha,
          cusp::device_memory)
{
    CUSP_PROFILE_SCOPED();
    throw cusp::not_implemented_exception("CUBLAS fill not implemented");
}

template <typename Array>
void fill(const Array& x,
          typename Array::value_type alpha,
          cusp::device_memory)
{
    CUSP_PROFILE_SCOPED();
    throw cusp::not_implemented_exception("CUBLAS fill not implemented");
}

template <typename Array>
typename norm_type<typename Array::value_type>::type
nrm1(const Array& x, cusp::device_memory)
{
    CUSP_PROFILE_SCOPED();
    throw cusp::not_implemented_exception("CUBLAS nrm1 not implemented");

    return 0;
}

template <typename Array>
typename norm_type<typename Array::value_type>::type
nrm2(const Array& x, cusp::device_memory)
{
    CUSP_PROFILE_SCOPED();
    throw cusp::not_implemented_exception("CUBLAS nrm2 not implemented");

    return 0;
}

template <typename Array>
typename Array::value_type
nrmmax(const Array& x, cusp::device_memory)
{
    CUSP_PROFILE_SCOPED();
    throw cusp::not_implemented_exception("CUBLAS nrmmax not implemented");

    return 0;
}

template <typename Array>
void scal(Array& x,
          typename Array::value_type alpha,
          cusp::device_memory)
{
    CUSP_PROFILE_SCOPED();
    throw cusp::not_implemented_exception("CUBLAS scal not implemented");
}

template <typename Array>
void scal(const Array& x,
          typename Array::value_type alpha,
          cusp::device_memory)
{
    CUSP_PROFILE_SCOPED();
    throw cusp::not_implemented_exception("CUBLAS scal not implemented");
}

template<typename Array2d1, typename Array1d1, typename Array1d2>
void gemv(const Array2d1& A,
          const Array1d1& x,
          Array1d2& y,
          cusp::device_memory)
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

    result = detail::gemv(__cublas.handle, trans, m, n, &alpha,
                          A_p, m, x_p, 1, &beta, y_p, 1);

    if(result != CUBLAS_STATUS_SUCCESS)
        throw cusp::runtime_exception("CUBLAS gemv failed!");
}

template<typename Array2d1, typename Array2d2, typename Array2d3>
void gemm(Array2d1& A,
          Array2d2& B,
          Array2d3& C,
          cusp::device_memory)
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

    result = detail::gemm(__cublas.handle, transa, transb,
                          m, n, k, &alpha, A_p, m,
                          B_p, k, &beta, C_p, m);

    if(result != CUBLAS_STATUS_SUCCESS)
        throw cusp::runtime_exception("CUBLAS gemm failed!");
}

} // end namespace cublas
} // end namespace blas
} // end namespace cusp

