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

#include <cusp/detail/device/generalized_spmv/coo_flat.h>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/find.h>
#include <thrust/scan.h>

#include <thrust/iterator/constant_iterator.h>

namespace cusp
{
namespace detail
{
namespace device
{

template <typename KeyType>
struct is_first_functor
{
  typedef bool result_type;

  is_first_functor(){}

  template <typename Tuple>
    __host__ __device__
  bool operator()(const Tuple& t) const
  {
    const KeyType i = thrust::get<0>(t);
    const KeyType j = thrust::get<1>(t);

    return i != j;
  }
};


template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction>
void reduce_by_key(InputIterator1 keys_first, 
                   InputIterator1 keys_last,
                   InputIterator2 values_first,
                   OutputIterator1 keys_output,
                   OutputIterator2 values_output,
                   BinaryPredicate binary_pred,
                   BinaryFunction binary_op)
{
    CUSP_PROFILE_SCOPED();

    typedef unsigned int IndexType;
    typedef typename thrust::iterator_traits<InputIterator1>::value_type  KeyType;
    typedef typename thrust::iterator_traits<OutputIterator2>::value_type ValueType;

    // input size
    IndexType n = keys_last - keys_first;

    cusp::array1d<IndexType,cusp::device_memory> row_indices(n);
    row_indices[0] = 0;

    // mark first element in each group
    thrust::binary_negate<BinaryPredicate> not_binary_pred(binary_pred);

	// TODO : is_first_functor should use take the not_binary_pred predicate as an argument
	// scan the predicates
	thrust::inclusive_scan(thrust::make_transform_iterator( thrust::make_zip_iterator(thrust::make_tuple(keys_first, keys_first+1)), is_first_functor<KeyType>() ),
                           thrust::make_transform_iterator( thrust::make_zip_iterator(thrust::make_tuple(keys_last-1, keys_last))  , is_first_functor<KeyType>() ),
                           row_indices.begin()+1);

	size_t num_rows = row_indices.back() + 1;

	cusp::detail::device::cuda::spmv_coo(num_rows,
					     row_indices.size(),
					     row_indices.begin(),
					     thrust::constant_iterator<IndexType>(0),
					     values_first,
					     thrust::constant_iterator<ValueType>(1),
					     thrust::constant_iterator<ValueType>(0),
					     values_output,
					     thrust::multiplies<ValueType>(),
					     binary_op);

    // copy the first element before scatter
    keys_output[0] = keys_first[0];

    // scatter first elements
    thrust::scatter_if
        (keys_first + 1, keys_last, row_indices.begin() + 1,
         thrust::make_transform_iterator( thrust::make_zip_iterator(thrust::make_tuple(keys_first, keys_first+1))  , 
                                          is_first_functor<KeyType>() ),
         keys_output);
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

