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

#include <thrust/extrema.h>

#include <cusp/system/cuda/arch.h>
#include <cusp/system/cuda/utils.h>

#include <thrust/device_ptr.h>

#include <algorithm>

namespace cusp
{
namespace system
{
namespace cuda
{

////////////////////////////////////////////////////////////////////////
// DIA SpMV kernels
///////////////////////////////////////////////////////////////////////
//
// Diagonal matrices arise in grid-based discretizations using stencils.
// For instance, the standard 5-point discretization of the two-dimensional
// Laplacian operator has the stencil:
//      [  0  -1   0 ]
//      [ -1   4  -1 ]
//      [  0  -1   0 ]
// and the resulting DIA format has 5 diagonals.
//
// spmv_dia
//   Each thread computes y[i] += A[i,:] * x
//   (the dot product of the i-th row of A with the x vector)
//
// spmv_dia_tex
//   Same as spmv_dia, except x is accessed via texture cache.
//


template <typename IndexType, typename ValueType, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE,1)
__global__ void
spmv_dia_kernel(const IndexType num_rows,
                const IndexType num_cols,
                const IndexType num_diagonals,
                const IndexType pitch,
                const IndexType * diagonal_offsets,
                const ValueType * values,
                const ValueType * x,
                ValueType * y)
{
    __shared__ IndexType offsets[BLOCK_SIZE];

    const IndexType thread_id = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const IndexType grid_size = BLOCK_SIZE * gridDim.x;

    for(IndexType base = 0; base < num_diagonals; base += BLOCK_SIZE)
    {
        // read a chunk of the diagonal offsets into shared memory
        const IndexType chunk_size = thrust::min(IndexType(BLOCK_SIZE), num_diagonals - base);

        if(threadIdx.x < chunk_size)
            offsets[threadIdx.x] = diagonal_offsets[base + threadIdx.x];

        __syncthreads();

        // process chunk
        for(IndexType row = thread_id; row < num_rows; row += grid_size)
        {
            ValueType sum = (base == 0) ? ValueType(0) : y[row];

            // index into values array
            IndexType idx = row + pitch * base;

            for(IndexType n = 0; n < chunk_size; n++)
            {
                const IndexType col = row + offsets[n];

                if(col >= 0 && col < num_cols)
                {
                    const ValueType A_ij = values[idx];
                    sum += A_ij * x[col];
                }

                idx += pitch;
            }

            y[row] = sum;
        }

        // wait until all threads are done reading offsets
        __syncthreads();
    }
}


template <typename DerivedPolicy,
         typename MatrixType,
         typename VectorType1,
         typename VectorType2,
         typename UnaryFunction,
         typename BinaryFunction1,
         typename BinaryFunction2>
void multiply(cuda::execution_policy<DerivedPolicy>& exec,
              MatrixType& A,
              VectorType1& x,
              VectorType2& y,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce,
              dia_format,
              array1d_format,
              array1d_format)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;

    const size_t BLOCK_SIZE = 256;
    const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(spmv_dia_kernel<IndexType, ValueType, BLOCK_SIZE>, BLOCK_SIZE, (size_t) sizeof(IndexType) * BLOCK_SIZE);
    const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, DIVIDE_INTO(A.num_rows, BLOCK_SIZE));

    const IndexType num_diagonals = A.values.num_cols;
    const IndexType pitch         = A.values.pitch;

    // TODO can this be removed?
    if (num_diagonals == 0)
    {
        // empty matrix
        thrust::fill(exec, y.begin(), y.begin() + A.num_rows, ValueType(0));
        return;
    }

    const IndexType * D = thrust::raw_pointer_cast(&A.diagonal_offsets[0]);
    const ValueType * V = thrust::raw_pointer_cast(&A.values(0,0));
    const ValueType * x_ptr = thrust::raw_pointer_cast(&x[0]);
    ValueType * y_ptr = thrust::raw_pointer_cast(&y[0]);

    cudaStream_t s = stream(thrust::detail::derived_cast(exec));

    spmv_dia_kernel<IndexType, ValueType, BLOCK_SIZE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, s>>>
    (A.num_rows, A.num_cols, num_diagonals, pitch, D, V, x_ptr, y_ptr);
}

} // end namespace cuda
} // end namespace system
} // end namespace cusp


