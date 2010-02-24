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

#include <cusp/coo_matrix.h>
#include <cusp/array2d.h>

#include <cusp/detail/functional.h>
#include <cusp/detail/generic/multiply.h>
#include <cusp/detail/host/spmv.h>

namespace cusp
{
namespace detail
{
namespace host
{

//////////////////////////////////
// Matrix-Matrix Multiplication //
//////////////////////////////////
template <typename IndexType,
          typename ValueType,
          typename MemorySpace>
void multiply(const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& A,
              const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& B,
                    cusp::coo_matrix<IndexType,ValueType,MemorySpace>& C)
{
    cusp::detail::generic::multiply(A,B,C);
}
    
template <typename ValueType,
          typename MemorySpace>
void multiply(const cusp::array2d<ValueType,MemorySpace>& A,
              const cusp::array2d<ValueType,MemorySpace>& B,
                    cusp::array2d<ValueType,MemorySpace>& C)
{
    cusp::detail::generic::multiply(A,B,C);
}

//////////////////////////////////
// Matrix-Vector Multiplication //
//////////////////////////////////
template <typename IndexType,
          typename ValueType,
          typename MemorySpace1,
          typename MemorySpace2,
          typename MemorySpace3>
void multiply(const cusp::coo_matrix<IndexType,ValueType,MemorySpace1>& A,
              const cusp::array1d<ValueType,MemorySpace2>& B,
                    cusp::array1d<ValueType,MemorySpace3>& C)
{
    cusp::detail::host::spmv_coo
        (A.num_rows, A.num_cols, A.num_entries,
         &A.row_indices[0], &A.column_indices[0], &A.values[0],
         &B[0], &C[0],
         cusp::detail::zero_function<ValueType>(), thrust::multiplies<ValueType>(), thrust::plus<ValueType>());
}

template <typename IndexType,
          typename ValueType,
          typename MemorySpace1,
          typename MemorySpace2,
          typename MemorySpace3>
void multiply(const cusp::csr_matrix<IndexType,ValueType,MemorySpace1>& A,
              const cusp::array1d<ValueType,MemorySpace2>& B,
                    cusp::array1d<ValueType,MemorySpace3>& C)
{
    cusp::detail::host::spmv_csr
        (A.num_rows, A.num_cols,
         &A.row_offsets[0], &A.column_indices[0], &A.values[0],
         &B[0], &C[0],
         cusp::detail::zero_function<ValueType>(), thrust::multiplies<ValueType>(), thrust::plus<ValueType>());
}

template <typename IndexType,
          typename ValueType,
          typename MemorySpace1,
          typename MemorySpace2,
          typename MemorySpace3>
void multiply(const cusp::dia_matrix<IndexType,ValueType,MemorySpace1>& A,
              const cusp::array1d<ValueType,MemorySpace2>& B,
                    cusp::array1d<ValueType,MemorySpace3>& C)
{
    const IndexType num_diagonals = A.values.num_cols;
    const IndexType stride        = A.values.num_rows;

    cusp::detail::host::spmv_dia
        (A.num_rows, A.num_cols, num_diagonals, stride,
         &A.diagonal_offsets[0], &A.values.values[0],
         &B[0], &C[0],
         cusp::detail::zero_function<ValueType>(), thrust::multiplies<ValueType>(), thrust::plus<ValueType>());
}

template <typename IndexType,
          typename ValueType,
          typename MemorySpace1,
          typename MemorySpace2,
          typename MemorySpace3>
void multiply(const cusp::ell_matrix<IndexType,ValueType,MemorySpace1>& A,
              const cusp::array1d<ValueType,MemorySpace2>& B,
                    cusp::array1d<ValueType,MemorySpace3>& C)
{
    const IndexType stride              = A.column_indices.num_rows;
    const IndexType num_entries_per_row = A.column_indices.num_cols;

    cusp::detail::host::spmv_ell
        (A.num_rows, A.num_cols, num_entries_per_row, stride,
         &A.column_indices.values[0], &A.values.values[0],
         &B[0], &C[0],
         cusp::detail::zero_function<ValueType>(), thrust::multiplies<ValueType>(), thrust::plus<ValueType>());
}

template <typename IndexType,
          typename ValueType,
          typename MemorySpace1,
          typename MemorySpace2,
          typename MemorySpace3>
void multiply(const cusp::hyb_matrix<IndexType,ValueType,MemorySpace1>& A,
              const cusp::array1d<ValueType,MemorySpace2>& B,
                    cusp::array1d<ValueType,MemorySpace3>& C)
{
    const IndexType stride              = A.ell.column_indices.num_rows;
    const IndexType num_entries_per_row = A.ell.column_indices.num_cols;

    cusp::detail::host::spmv_ell
        (A.num_rows, A.num_cols, num_entries_per_row, stride,
         &A.ell.column_indices.values[0], &A.ell.values.values[0],
         &B[0], &C[0],
         cusp::detail::zero_function<ValueType>(), thrust::multiplies<ValueType>(), thrust::plus<ValueType>());
    
    cusp::detail::host::spmv_coo
        (A.coo.num_rows, A.coo.num_cols, A.coo.num_entries,
         &A.coo.row_indices[0], &A.coo.column_indices[0], &A.coo.values[0],
         &B[0], &C[0],
         thrust::identity<ValueType>(), thrust::multiplies<ValueType>(), thrust::plus<ValueType>());
}

template <typename ValueType,
          typename MemorySpace1,
          typename MemorySpace2,
          typename MemorySpace3>
void multiply(const cusp::array2d<ValueType,MemorySpace1>& A,
              const cusp::array1d<ValueType,MemorySpace2>& B,
                    cusp::array1d<ValueType,MemorySpace3>& C)
{
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

} // end namespace host
} // end namespace detail
} // end namespace cusp

