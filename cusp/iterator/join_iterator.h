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

/*! \file cusp/iterator/strided_iterator.h
 *  \brief An iterator which returns strided access to array entries
 */

#pragma once
#include <thrust/functional.h>

#include <thrust/iterator/transform_iterator.h>
namespace cusp
{

/*! \addtogroup iterators Iterators
 *  \ingroup iterators
 *  \brief Various customized Thrust based iterators
 *  \{
 */

/*! \brief RandomAccessIterator for strided access to array entries.
 *
 * \tparam RandomAccessIterator The iterator type used to encapsulate the underlying data.
 *
 * \par Overview
 * \p strided_iterator is an iterator which represents a pointer into
 *  a strided range entries in a underlying array. This iterator is useful
 *  for creating a strided sublist of entries from a larger iterator.
 *
 * \par Example
 *  The following code snippet demonstrates how to create a \p strided_iterator whose
 *  \c value_type is \c int and whose values are gather from a \p counting_array.
 *
 *  \code
 *  #include <cusp/array1d.h>
 *  #include <cusp/iterator/join_iterator.h>
 *
 *  #include <iostream>
 *
 *  int main(void)
 *  {
 *    typedef cusp::counting_iterator<int>                                           CountingIterator;
 *    typedef cusp::constant_iterator<int>                                           ConstantIterator;
 *    typedef cusp::array1d<int,cusp::device_memory>::iterator                       ArrayIterator;
 *    typedef cusp::join_iterator<CountingIterator,ConstantIterator,ArrayIterator> ConcatIterator
 *
 *    CountingIterator a(4);
 *    ConstantIterator b(10);
 *    cusp::array1d<int,cusp::device_memory> indices(a.size() + b.size());
 *    thrust::sequence(indices.begin(), indices.end());
 *    ConcatIterator iter(a.begin(), a.end(), b.begin(), b.end(), indices.begin());
 *
 *    std::cout << iter[0] << std::endl;   // returns 0
 *    std::cout << iter[3] << std::endl;   // returns 3
 *    std::cout << iter[4] << std::endl;   // returns 10
 *
 *    return 0;
 *  }
 *  \endcode
 */
template <typename Iterator1, typename Iterator2, typename IndexIterator>
class join_iterator
{
    public:

    typedef typename thrust::iterator_value<Iterator1>::type      value_type;
    typedef typename thrust::iterator_difference<Iterator1>::type difference_type;

    struct join_select_functor : public thrust::unary_function<difference_type,value_type>
    {
        Iterator1 first;
        Iterator2 second;
        difference_type first_size;

        join_select_functor(Iterator1 first, Iterator2 second, difference_type first_size)
            : first(first), second(second-first_size), first_size(first_size) {}

        __host__ __device__
        value_type operator()(const difference_type& i) const
        {
            return i < first_size ? first[i] : second[i];
        }
    };

    typedef typename thrust::transform_iterator<concat_select_functor, IndexIterator> TransformIterator;

    // type of the join_iterator
    typedef TransformIterator iterator;

    // construct join_iterator using first_begin and second_begin
    join_iterator(Iterator1 first_begin, Iterator1 first_end, Iterator2 second_begin, Iterator2 second_end, IndexIterator indices_begin)
        : first_begin(first_begin), first_end(first_end), second_begin(second_begin), second_end(second_end), indices_begin(indices_begin) {}

    iterator begin(void) const
    {
        return TransformIterator(indices_begin, concat_select_functor(first_begin, second_begin, first_end-first_begin));
    }

    iterator end(void) const
    {
        return begin() + (first_end-first_begin) + (second_end-second_begin);
    }

    protected:
    IndexIterator indices_begin;

    Iterator1 first_begin;
    Iterator1 first_end;
    Iterator2 second_begin;
    Iterator2 second_end;
};

/*! \} // end iterators
 */

} // end namespace cusp

