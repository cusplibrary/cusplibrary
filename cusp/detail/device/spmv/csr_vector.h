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

#include <cusp/csr_matrix.h>

#include <cusp/detail/device/common.h>
#include <cusp/detail/device/utils.h>
#include <cusp/detail/device/texture.h>

#include <thrust/experimental/arch.h>

namespace cusp
{
namespace detail
{
namespace device
{

//////////////////////////////////////////////////////////////////////////////
// CSR SpMV kernels based on a vector model (one warp per row)
//////////////////////////////////////////////////////////////////////////////
//
// spmv_csr_vector_device
//   Each row of the CSR matrix is assigned to a warp.  The warp computes
//   y[i] = A[i,:] * x, i.e. the dot product of the i-th row of A with 
//   the x vector, in parallel.  This division of work implies that 
//   the CSR index and data arrays (Aj and Ax) are accessed in a contiguous
//   manner (but generally not aligned).  On GT200 these accesses are
//   coalesced, unlike kernels based on the one-row-per-thread division of 
//   work.  Since an entire 32-thread warp is assigned to each row, many 
//   threads will remain idle when their row contains a small number 
//   of elements.  This code relies on implicit synchronization among 
//   threads in a warp.
//
// spmv_csr_vector_tex_device
//   Same as spmv_csr_vector_tex_device, except that the texture cache is 
//   used for accessing the x vector.

template <typename IndexType, typename ValueType, unsigned int BLOCK_SIZE, bool UseCache>
__global__ void
spmv_csr_vector_kernel(const IndexType num_rows,
                       const IndexType * Ap, 
                       const IndexType * Aj, 
                       const ValueType * Ax, 
                       const ValueType * x, 
                             ValueType * y)
{
    __shared__ ValueType sdata[BLOCK_SIZE + 16];                          // padded to avoid reduction ifs
    __shared__ IndexType ptrs[BLOCK_SIZE/WARP_SIZE][2];
    
    const IndexType thread_id   = BLOCK_SIZE * blockIdx.x + threadIdx.x;  // global thread index
    const IndexType thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
    const IndexType warp_id     = thread_id   / WARP_SIZE;                // global warp index
    const IndexType warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
    const IndexType num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;   // total number of active warps

    for(IndexType row = warp_id; row < num_rows; row += num_warps)
    {
        // use two threads to fetch Ap[row] and Ap[row+1]
        // this is considerably faster than the straightforward version
        if(thread_lane < 2)
            ptrs[warp_lane][thread_lane] = Ap[row + thread_lane];
        const IndexType row_start = ptrs[warp_lane][0];                   //same as: row_start = Ap[row];
        const IndexType row_end   = ptrs[warp_lane][1];                   //same as: row_end   = Ap[row+1];

        // compute local sum
        ValueType sum = 0;
        for(IndexType jj = row_start + thread_lane; jj < row_end; jj += WARP_SIZE)
            sum += Ax[jj] * fetch_x<UseCache>(Aj[jj], x);

        // reduce local sums to row sum (ASSUME: warpsize 32)
        sdata[threadIdx.x] = sum;
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x + 16];
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  8];
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  4];
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  2];
        sdata[threadIdx.x] = sum = sum + sdata[threadIdx.x +  1];
       
//// Alternative method (slightly slower)
//        // compute local sum
//        sdata[threadIdx.x] = 0;
//        for(IndexType jj = row_start + thread_lane; jj < row_end; jj += WARP_SIZE)
//            sdata[threadIdx.x] += Ax[jj] * fetch_x<UseCache>(Aj[jj], x);
//
//        // reduce local sums to row sum (ASSUME: warpsize 32)
//        sdata[threadIdx.x] += sdata[threadIdx.x + 16];
//        sdata[threadIdx.x] += sdata[threadIdx.x +  8];
//        sdata[threadIdx.x] += sdata[threadIdx.x +  4];
//        sdata[threadIdx.x] += sdata[threadIdx.x +  2];
//        sdata[threadIdx.x] += sdata[threadIdx.x +  1];

        // first thread writes warp result
        if (thread_lane == 0)
            y[row] += sdata[threadIdx.x];
    }
}

template <bool UseCache, typename IndexType, typename ValueType>
void __spmv_csr_vector(const csr_matrix<IndexType,ValueType,cusp::device>& csr, 
                       const ValueType * x, 
                             ValueType * y)
{
    const unsigned int BLOCK_SIZE = 128;
    const unsigned int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
    const unsigned int MAX_BLOCKS = thrust::experimental::arch::max_active_blocks(spmv_csr_vector_kernel<IndexType, ValueType, BLOCK_SIZE, UseCache>, BLOCK_SIZE, (size_t) 0);
    const unsigned int NUM_BLOCKS = std::min(MAX_BLOCKS, DIVIDE_INTO(csr.num_rows, WARPS_PER_BLOCK));
    
    if (UseCache)
        bind_x(x);

    spmv_csr_vector_kernel<IndexType, ValueType, BLOCK_SIZE, UseCache> <<<NUM_BLOCKS, BLOCK_SIZE>>> 
        (csr.num_rows,
         thrust::raw_pointer_cast(&csr.row_offsets[0]),
         thrust::raw_pointer_cast(&csr.column_indices[0]),
         thrust::raw_pointer_cast(&csr.values[0]),
         x, y);

    if (UseCache)
        unbind_x(x);
}

template <typename IndexType, typename ValueType>
void spmv_csr_vector(const cusp::csr_matrix<IndexType,ValueType,cusp::device>& csr, 
                     const ValueType * x, 
                           ValueType * y)
{
    __spmv_csr_vector<false>(csr, x, y);
}

template <typename IndexType, typename ValueType>
void spmv_csr_vector_tex(const csr_matrix<IndexType,ValueType,cusp::device>& csr, 
                         const ValueType * x, 
                               ValueType * y)
{
    __spmv_csr_vector<true>(csr, x, y);
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

