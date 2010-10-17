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

#include <cusp/detail/device/arch.h>
#include <cusp/detail/device/utils.h>

#include <thrust/transform.h>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/device/dereference.h>

namespace cusp
{
namespace detail
{
namespace device
{
namespace cuda
{

template <int BLOCK_SIZE,
          typename SizeType,
          typename IndexIterator1,
          typename IndexIterator2,
          typename ValueIterator1,
          typename ValueIterator2,
          typename ValueIterator4,
          typename BinaryFunction1,
          typename BinaryFunction2>
__launch_bounds__(BLOCK_SIZE,1)
__global__
void spmv_coo_kernel(SizeType        num_entries,
                     IndexIterator1  row_indices,
                     IndexIterator2  column_indices,
                     ValueIterator1  values,
                     ValueIterator2  x, 
                     ValueIterator4  z,
                     BinaryFunction1 combine,
                     BinaryFunction2 reduce)
{
  typedef typename thrust::iterator_value<IndexIterator1>::type IndexType1;
  typedef typename thrust::iterator_value<IndexIterator2>::type IndexType2;
  typedef typename thrust::iterator_value<ValueIterator1>::type ValueType1;
  typedef typename thrust::iterator_value<ValueIterator2>::type ValueType2;
  typedef typename thrust::iterator_value<ValueIterator4>::type ValueType4;

  if (threadIdx.x == 0)
  {
    for (SizeType i = 0; i < num_entries; i++)
    {
      IndexType1 i   = thrust::detail::device::dereference(row_indices);
      IndexType2 j   = thrust::detail::device::dereference(column_indices);
      ValueType1 Aij = thrust::detail::device::dereference(values);

      ValueIterator2 tmp_x = x + j;  ValueType2 xj = thrust::detail::device::dereference(tmp_x);
      ValueIterator4 tmp_z = z + i;  ValueType4 zi = thrust::detail::device::dereference(tmp_z);

      thrust::detail::device::dereference(tmp_z) = reduce(zi, combine(Aij, xj));

      ++row_indices;
      ++column_indices;
      ++values;
    }
  }
}

template <typename SizeType,
          typename IndexIterator1,
          typename IndexIterator2,
          typename ValueIterator1,
          typename ValueIterator2,
          typename ValueIterator3,
          typename ValueIterator4,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void spmv_coo(SizeType        num_rows,
              SizeType        num_entries,
              IndexIterator1  row_indices,
              IndexIterator2  column_indices,
              ValueIterator1  values,
              ValueIterator2  x, 
              ValueIterator3  y,
              ValueIterator4  z,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce)
{
  const SizeType block_size = 256;
  //const SizeType max_blocks = cusp::detail::device::arch::max_active_blocks(spmv_coo_scalar_kernel<block_size, SizeType, IndexIterator1, IndexIterator2, ValueIterator1, ValueIterator2, ValueIterator4, BinaryFunction1, BinaryFunction2>, block_size, (size_t) 0);
  const SizeType num_blocks = 1; //std::min(max_blocks, DIVIDE_INTO(num_entries, block_size));

  thrust::transform(y, y + num_rows, z, initialize);

  if (num_entries == 0) return;

  // note: we don't need to pass y or initialize
  spmv_coo_kernel<block_size><<<num_blocks, block_size>>>
    (num_entries,
     row_indices, column_indices, values,
     x, z,
     combine, reduce);
}

} // end namespace cuda
} // end namespace device
} // end namespace detail
} // end namespace cusp

