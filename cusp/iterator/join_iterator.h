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

/*! \file cusp/iterator/join_iterator.h
 *  \brief An iterator which concatenates two separate iterators.
 */

#pragma once

#include <cusp/detail/config.h>

#include <thrust/distance.h>
#include <thrust/functional.h>

#include <thrust/iterator/transform_iterator.h>

namespace cusp
{

/*! \addtogroup iterators Iterators
 *  \ingroup iterators
 *  \brief Various customized Thrust based iterators
 *  \{
 */

/*! \brief RandomAccessIterator for access to array entries from two
 * concatenated iterators.
 *
 * \tparam Iterator1 The iterator type used to encapsulate the first set of
 * entries.
 * \tparam Iterator2 The iterator type used to encapsulate the second set of
 * entries.
 * \tparam IndexIterator The iterator type used to order concatenated entries
 * from two separate iterators.
 *
 * \par Overview
 * \p join_iterator is an iterator which represents a pointer into
 *  a concatenated range of entries from two underlying arrays. This iterator
 *  is useful for creating a single range of permuted entries from two
 *  different iterators.
 *
 * \par Example
 *  The following code snippet demonstrates how to create a \p join_iterator whose
 *  \c value_type is \c int and whose values are gather from a \p counting_iterator
 *  and a \p constant_iterator.
 *
 *  \code
 *  #include <cusp/array1d.h>
 *  #include <cusp/iterator/join_iterator.h>
 *
 *  #include <thrust/sequence.h>
 *
 *  #include <iostream>
 *
 *  int main(void)
 *  {
 *    typedef cusp::counting_array<int>                                              CountingArray;
 *    typedef cusp::constant_array<int>                                              ConstantArray;
 *    typedef typename CountingArray::iterator                                       CountingIterator;
 *    typedef typename ConstantArray::iterator                                       ConstantIterator;
 *    typedef cusp::array1d<int,cusp::device_memory>::iterator                       ArrayIterator;
 *    typedef cusp::join_iterator<CountingIterator,ConstantIterator,ArrayIterator>   JoinIterator;
 *
 *    // a = [0, 1, 2, 3]
 *    CountingArray a(4);
 *    // b = [10, 10, 10, 10, 10]
 *    ConstantArray b(5, 10);
 *    cusp::array1d<int,cusp::device_memory> indices(a.size() + b.size());
 *    // set indices to a sequence for simple in order access
 *    thrust::sequence(indices.begin(), indices.end());
 *    // iter = [0, 1, 2, 3, 10, 10, 10, 10, 10]
 *    JoinIterator iter(a.begin(), a.end(), b.begin(), b.end(), indices.begin());
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

    /*! \cond */
    typedef typename thrust::iterator_value<Iterator1>::type                       value_type;
    typedef typename thrust::iterator_system<Iterator1>::type                      memory_space;
    typedef typename thrust::iterator_pointer<Iterator1>::type                     pointer;
    typedef typename thrust::iterator_reference<Iterator1>::type                   reference;
    typedef typename thrust::iterator_difference<Iterator1>::type                  difference_type;
    typedef typename thrust::iterator_difference<Iterator1>::type                  size_type;

    struct join_select_functor : public thrust::unary_function<difference_type,value_type>
    {
        Iterator1 first;
        Iterator2 second;
        difference_type first_size;

        __host__ __device__
        join_select_functor(void){}

        __host__ __device__
        join_select_functor(Iterator1 first, Iterator2 second, difference_type first_size)
            : first(first), second(second-first_size), first_size(first_size) {}

        __host__ __device__
        value_type operator()(const difference_type& i) const
        {
            return i < first_size ? first[i] : second[i];
        }
    };

    typedef typename thrust::transform_iterator<join_select_functor, IndexIterator> TransformIterator;
    /*! \endcond */

    // type of the join_iterator
    typedef TransformIterator iterator;

    /*! \brief This constructor builds a \p join_iterator from two iterators.
     *  \param first_begin The beginning of the first range.
     *  \param first_end The end of the first range.
     *  \param second_begin The beginning of the second range.
     *  \param second_end The end of the second range.
     *  \param indices_begin The permutation indices used to order entries
     *  from the two joined iterators.
     */
    join_iterator(Iterator1 first_begin, Iterator1 first_end,
                  Iterator2 second_begin, Iterator2 second_end,
                  IndexIterator indices_begin)
        : first_begin(first_begin), first_end(first_end),
          second_begin(second_begin), second_end(second_end),
          indices_begin(indices_begin) {}

    /*! \brief This method returns an iterator pointing to the beginning of
     *  this joined sequence of permuted entries.
     *  \return mStart
     */
    iterator begin(void) const
    {
        return TransformIterator(indices_begin, join_select_functor(first_begin, second_begin, thrust::distance(first_begin,first_end)));
    }

    /*! \brief This method returns an iterator pointing to one element past
     *  the last of this joined sequence of permuted entries.
     *  \return mEnd
     */
    iterator end(void) const
    {
        return begin() + thrust::distance(first_begin,first_end) + thrust::distance(second_begin,second_end);
    }

    /*! \brief Subscript access to the data contained in this iterator.
     *  \param n The index of the element for which data should be accessed.
     *  \return Read/write reference to data.
     *
     *  This operator allows for easy, array-style, data access.
     *  Note that data access with this operator is unchecked and
     *  out_of_range lookups are not defined.
     */
    reference operator[](size_type n) const
    {
        return *(begin() + n);
    }

    protected:

    /*! \cond */
    Iterator1 first_begin;
    Iterator1 first_end;
    Iterator2 second_begin;
    Iterator2 second_end;

    IndexIterator indices_begin;
    /*! \cond */
};

/*! \} // end iterators
 */

} // end namespace cusp

