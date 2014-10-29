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

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/functional.h>

namespace cusp
{

template<typename T>
struct stride_functor : public thrust::unary_function<T,T>
{
    const T stride;

    stride_functor(T stride)
        : stride(stride) {}

    __host__ __device__
    T operator()(const T& i) const
    {
        return stride * i;
    }
};

/*! \addtogroup iterators Iterators
 *  \ingroup iterators
 *  \brief Various customized Thrust based iterators
 *  \{
 */

template <typename RandomAccessIterator>
class strided_iterator
{
public:

    typedef typename thrust::iterator_value<RandomAccessIterator>::type                       value_type;
    typedef typename thrust::iterator_system<RandomAccessIterator>::type                      memory_space;
    typedef typename thrust::iterator_pointer<RandomAccessIterator>::type                     pointer;
    typedef typename thrust::iterator_reference<RandomAccessIterator>::type                   reference;
    typedef typename thrust::iterator_difference<RandomAccessIterator>::type                  difference_type;
    typedef typename thrust::iterator_difference<RandomAccessIterator>::type                  size_type;

    typedef stride_functor<difference_type>                                                   Functor;
    typedef typename thrust::counting_iterator<difference_type>                               CountingIterator;
    typedef typename thrust::transform_iterator<Functor, CountingIterator>                    TransformIterator;
    typedef typename thrust::permutation_iterator<RandomAccessIterator,TransformIterator>     PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    strided_iterator(RandomAccessIterator first, RandomAccessIterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}

    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), Functor(stride)));
    }

    iterator end(void) const
    {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }

protected:
    RandomAccessIterator first;
    RandomAccessIterator last;
    difference_type stride;
}; // end strided_iterator

/*! \} // end iterators
 */

} // end namespace cusp

