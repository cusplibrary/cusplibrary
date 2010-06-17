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

#include <cusp/detail/matrix_traits.h>
#include <cusp/detail/functional.h>

#include <cusp/detail/host/spmv.h>
#include <cusp/detail/host/spmm/coo.h>
#include <cusp/detail/host/spmm/csr.h>

namespace cusp
{
namespace detail
{
namespace host
{

//////////////////////////////////
// Dense Matrix-Vector Multiply //
//////////////////////////////////
template <typename Matrix,
          typename Vector1,
          typename Vector2>
void multiply(const Matrix&  A,
              const Vector1& B,
                    Vector2& C,
              cusp::detail::array2d_format_tag,
              cusp::detail::array1d_format_tag,
              cusp::detail::array1d_format_tag)
{
    typedef typename Vector2::value_type ValueType;

    for(size_t i = 0; i < A.num_rows; i++)
    {
        ValueType sum = 0;
        for(size_t j = 0; j < A.num_cols; j++)
        {
            sum += A(i,j) * B[j];
        }
        C[i] = sum;
    }
}

///////////////////////////////////
// Sparse Matrix-Vector Multiply //
///////////////////////////////////
template <typename Matrix,
          typename Vector1,
          typename Vector2>
void multiply(const Matrix&  A,
              const Vector1& B,
                    Vector2& C,
              cusp::detail::coo_format_tag,
              cusp::detail::array1d_format_tag,
              cusp::detail::array1d_format_tag)
{
    cusp::detail::host::spmv_coo(A, B, C);
}

template <typename Matrix,
          typename Vector1,
          typename Vector2>
void multiply(const Matrix&  A,
              const Vector1& B,
                    Vector2& C,
              cusp::detail::csr_format_tag,
              cusp::detail::array1d_format_tag,
              cusp::detail::array1d_format_tag)
{
    cusp::detail::host::spmv_csr(A, B, C);
}

template <typename Matrix,
          typename Vector1,
          typename Vector2>
void multiply(const Matrix&  A,
              const Vector1& B,
                    Vector2& C,
              cusp::detail::dia_format_tag,
              cusp::detail::array1d_format_tag,
              cusp::detail::array1d_format_tag)
{
    cusp::detail::host::spmv_dia(A, B, C);
}

template <typename Matrix,
          typename Vector1,
          typename Vector2>
void multiply(const Matrix&  A,
              const Vector1& B,
                    Vector2& C,
              cusp::detail::ell_format_tag,
              cusp::detail::array1d_format_tag,
              cusp::detail::array1d_format_tag)
{
    cusp::detail::host::spmv_ell(A, B, C);
}

template <typename Matrix,
          typename Vector1,
          typename Vector2>
void multiply(const Matrix&  A,
              const Vector1& B,
                    Vector2& C,
              cusp::detail::hyb_format_tag,
              cusp::detail::array1d_format_tag,
              cusp::detail::array1d_format_tag)
{
    typedef typename Vector2::value_type ValueType;

    cusp::detail::host::spmv_ell(A.ell, B, C);
    cusp::detail::host::spmv_coo(A.coo, B, C, thrust::identity<ValueType>(), thrust::multiplies<ValueType>(), thrust::plus<ValueType>());
}

////////////////////////////////////////
// Sparse Matrix-BlockVector Multiply //
////////////////////////////////////////
//// TODO implement this w/ repeated SpMVs and then specialize
//template <typename Matrix,
//          typename Vector1,
//          typename Vector2>
//void multiply(const Matrix&  A,
//              const Vector1& B,
//                    Vector2& C,
//              cusp::detail::sparse_format_tag,
//              cusp::detail::array2d_format_tag,
//              cusp::detail::array2d_format_tag)
//{
//}

////////////////////////////////////////
// Dense Matrix-Matrix Multiplication //
////////////////////////////////////////
template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void multiply(const Matrix1&  A,
              const Matrix2& B,
                    Matrix3& C,
              cusp::detail::array2d_format_tag,
              cusp::detail::array2d_format_tag,
              cusp::detail::array2d_format_tag)
{
    typedef typename Matrix3::value_type ValueType;

    C.resize(A.num_rows, B.num_cols);

    for(size_t i = 0; i < C.num_rows; i++)
    {
        for(size_t j = 0; j < C.num_cols; j++)
        {
            ValueType v = 0;

            for(size_t k = 0; k < A.num_cols; k++)
                v += A(i,k) * B(k,j);
            
            C(i,j) = v;
        }
    }
}

/////////////////////////////////////////
// Sparse Matrix-Matrix Multiplication //
/////////////////////////////////////////
template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void multiply(const Matrix1&  A,
              const Matrix2& B,
                    Matrix3& C,
              cusp::detail::coo_format_tag,
              cusp::detail::coo_format_tag,
              cusp::detail::coo_format_tag)
{
    cusp::detail::host::spmm_coo(A,B,C);
}

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void multiply(const Matrix1& A,
              const Matrix2& B,
                    Matrix3& C,
              cusp::detail::sparse_format_tag,
              cusp::detail::sparse_format_tag,
              cusp::detail::sparse_format_tag)
{
    cusp::detail::host::spmm_csr(A,B,C);
}
  
/////////////////
// Entry Point //
/////////////////
template <typename Matrix,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(const Matrix&  A,
              const MatrixOrVector1& B,
                    MatrixOrVector2& C)
{
    cusp::detail::host::multiply(A, B, C,
            typename cusp::detail::matrix_format<Matrix>::type(),
            typename cusp::detail::matrix_format<MatrixOrVector1>::type(),
            typename cusp::detail::matrix_format<MatrixOrVector2>::type());
}

} // end namespace host
} // end namespace detail
} // end namespace cusp

