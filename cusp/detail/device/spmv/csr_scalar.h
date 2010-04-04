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

#include <cusp/detail/device/utils.h>
#include <cusp/detail/device/texture.h>

#include <thrust/experimental/arch.h>

namespace cusp
{
namespace detail
{
namespace device
{

////////////////////////////////////////////////////////////////////////
// CSR SpMV kernels based on a scalar model (one thread per row)
///////////////////////////////////////////////////////////////////////
//
// spmv_csr_scalar_device
//   Straightforward translation of standard CSR SpMV to CUDA
//   where each thread computes y[i] = A[i,:] * x 
//   (the dot product of the i-th row of A with the x vector)
//
// spmv_csr_scalar_tex_device
//   Same as spmv_csr_scalar_device, except x is accessed via texture cache.
//

template <typename IndexType, typename ValueType, bool UseCache>
__global__ void
spmv_csr_scalar_kernel(const IndexType num_rows,
                       const IndexType * Ap, 
                       const IndexType * Aj, 
                       const ValueType * Ax, 
                       const ValueType * x, 
                             ValueType * y)
{
    const IndexType thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const IndexType grid_size = gridDim.x * blockDim.x;

    for(IndexType row = thread_id; row < num_rows; row += grid_size)
    {
        const IndexType row_start = Ap[row];
        const IndexType row_end   = Ap[row+1];
        
        ValueType sum = 0;
    
        for (IndexType jj = row_start; jj < row_end; jj++)
            sum += Ax[jj] * fetch_x<UseCache>(Aj[jj], x);       

        y[row] = sum;
    }
}

    
template <bool UseCache, typename IndexType, typename ValueType>
void __spmv_csr_scalar(const csr_matrix<IndexType,ValueType,cusp::device_memory>& csr, 
                       const ValueType * x, 
                             ValueType * y)
{
    const unsigned int BLOCK_SIZE = 256;
    const unsigned int MAX_BLOCKS = thrust::experimental::arch::max_active_blocks(spmv_csr_scalar_kernel<IndexType, ValueType, UseCache>, BLOCK_SIZE, (size_t) 0);
    const unsigned int NUM_BLOCKS = std::min(MAX_BLOCKS, DIVIDE_INTO(csr.num_rows, BLOCK_SIZE));
    
    if (UseCache)
        bind_x(x);

    spmv_csr_scalar_kernel<IndexType, ValueType, UseCache> <<<NUM_BLOCKS, BLOCK_SIZE>>> 
        (csr.num_rows,
         thrust::raw_pointer_cast(&csr.row_offsets[0]),
         thrust::raw_pointer_cast(&csr.column_indices[0]),
         thrust::raw_pointer_cast(&csr.values[0]),
         x, y);

    if (UseCache)
        unbind_x(x);
}

template <typename IndexType, typename ValueType>
void spmv_csr_scalar(const csr_matrix<IndexType,ValueType,cusp::device_memory>& csr, 
                     const ValueType * x, 
                           ValueType * y)
{
    __spmv_csr_scalar<false>(csr, x, y);
}

template <typename IndexType, typename ValueType>
void spmv_csr_scalar_tex(const csr_matrix<IndexType,ValueType,cusp::device_memory>& csr, 
                         const ValueType * x, 
                               ValueType * y)
{
    __spmv_csr_scalar<true>(csr, x, y);
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

