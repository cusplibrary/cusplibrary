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

#include <cusp/detail/device/utils.h>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/experimental/arch.h>

namespace cusp
{
namespace detail
{
namespace device
{
namespace cuda
{

    
template <typename SizeType,
          typename OffsetIterator,
          typename IndexIterator,
          typename ValueIterator,
          typename InputIterator,
          typename InitialIterator,
          typename OutputIterator,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
__global__
void spmv_csr_scalar_kernel(SizeType        num_rows,
                            OffsetIterator  row_offsets,
                            IndexIterator   column_indices,
                            ValueIterator   values,
                            InputIterator   x, 
                            InitialIterator y,
                            OutputIterator  z,
                            UnaryFunction   initialize,
                            BinaryFunction1 combine,
                            BinaryFunction2 reduce)
{
    typedef typename thrust::iterator_value<OffsetIterator>::type OffsetType;
    typedef typename thrust::iterator_value<IndexIterator>::type  IndexType;
    typedef typename thrust::iterator_value<ValueIterator>::type  ValueType;
    typedef typename thrust::iterator_value<InputIterator>::type  InputType;
    typedef typename thrust::iterator_value<OutputIterator>::type OutputType;

    const SizeType thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const SizeType grid_size = gridDim.x * blockDim.x;

    for(SizeType i = thread_id; i < num_rows; i += grid_size)
    {
        OffsetIterator r0 = row_offsets; r0 += i;      OffsetType row_start = thrust::detail::device::dereference(r0); // row_offsets[i]
        OffsetIterator r1 = row_offsets; r1 += i + 1;  OffsetType row_end   = thrust::detail::device::dereference(r1); // row_offsets[i + 1]

        InitialIterator y0 = y; y0 += i;  OutputType sum = initialize(thrust::detail::device::dereference(y0));        // initialize(y[i])
    
        for (IndexType jj = row_start; jj < row_end; jj++)
        {
            IndexIterator c0 = column_indices; c0 += jj;  IndexType j    = thrust::detail::device::dereference(c0);  // j    = column_indices[jj]
            ValueIterator v0 = values;         v0 += jj;  ValueType A_ij = thrust::detail::device::dereference(v0);  // A_ij = values[jj]
            InputIterator x0 = x;              x0 += j;   InputType x_j  = thrust::detail::device::dereference(x0);  // x_j  = x[j]

            sum = reduce(sum, combine(A_ij, x_j));                                                                   // sum += A_ij * x_j
        }

        OutputIterator z0 = z; z0 += i;  thrust::detail::device::dereference(z0) = sum;                               // z[i] = sum
    }
}

    
template <typename SizeType,
          typename OffsetIterator,
          typename IndexIterator,
          typename ValueIterator,
          typename InputIterator,
          typename InitialIterator,
          typename OutputIterator,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void spmv_csr_scalar(SizeType        num_rows,
                     OffsetIterator  row_offsets,
                     IndexIterator   column_indices,
                     ValueIterator   values,
                     InputIterator   x, 
                     InitialIterator y,
                     OutputIterator  z,
                     UnaryFunction   initialize,
                     BinaryFunction1 combine,
                     BinaryFunction2 reduce)
{
    const SizeType block_size = 256;
    const SizeType max_blocks = thrust::experimental::arch::max_active_blocks(spmv_csr_scalar_kernel<SizeType, OffsetIterator, IndexIterator, ValueIterator, InputIterator, InitialIterator, OutputIterator, UnaryFunction, BinaryFunction1, BinaryFunction2>, block_size, (size_t) 0);
    const SizeType num_blocks = std::min(max_blocks, DIVIDE_INTO(num_rows, block_size));
    
    spmv_csr_scalar_kernel<<<num_blocks, block_size>>> 
        (num_rows,
         row_offsets, column_indices, values,
         x, y, z,
         initialize, combine, reduce);
}

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace cusp

