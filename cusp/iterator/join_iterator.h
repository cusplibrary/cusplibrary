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
 * \tparam Iterator3 The iterator type used to order concatenated entries
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

template <int size, typename T>
struct constant_tuple {
    using thrust::detail;

    typedef
    thrust::tuple<typename eval_if<(size > 0),identity_<T>,identity_<thrust::null_type> >::type,
                  typename eval_if<(size > 1),identity_<T>,identity_<thrust::null_type> >::type,
                  typename eval_if<(size > 2),identity_<T>,identity_<thrust::null_type> >::type,
                  typename eval_if<(size > 3),identity_<T>,identity_<thrust::null_type> >::type,
                  typename eval_if<(size > 4),identity_<T>,identity_<thrust::null_type> >::type,
                  typename eval_if<(size > 5),identity_<T>,identity_<thrust::null_type> >::type,
                  typename eval_if<(size > 6),identity_<T>,identity_<thrust::null_type> >::type,
                  typename eval_if<(size > 7),identity_<T>,identity_<thrust::null_type> >::type,
                  typename eval_if<(size > 8),identity_<T>,identity_<thrust::null_type> >::type,
                  typename eval_if<(size > 9),identity_<T>,identity_<thrust::null_type> >::type> type;
};

template <typename Tuple>
class join_iterator
{
public:

    /*! \cond */
    typedef typename thrust::tuple_element<0,Tuple>::type          Iterator1;
    typedef typename thrust::iterator_value<Iterator1>::type       value_type;
    typedef typename thrust::iterator_pointer<Iterator1>::type     pointer;
    typedef typename thrust::iterator_reference<Iterator1>::type   reference;
    typedef typename thrust::iterator_difference<Iterator1>::type  difference_type;
    typedef typename thrust::iterator_difference<Iterator1>::type  size_type;
    typedef typename thrust::iterator_system<Iterator1>::type      memory_space;

    const static size_t tuple_size = thrust::tuple_size<Tuple>::value;

    // forward definition
    struct join_select_functor;

    typedef typename constant_tuple<tuple_size-1,size_t>::type            SizesTuple;
    typedef typename thrust::tuple_element<tuple_size-1,Tuple>::type      IndexIterator;
    typedef thrust::transform_iterator<join_select_functor,IndexIterator> TransformIterator;

    struct join_select_functor : public thrust::unary_function<difference_type,value_type>
    {
        SizesTuple t1;
        Tuple t2;

        __host__ __device__
        join_select_functor(const SizesTuple& t1, const Tuple& t2)
            : t1(t1), t2(t2) {}

        __host__ __device__
        value_type operator()(const difference_type& i)
        {
            return i < thrust::get<0>(t1) ? thrust::get<0>(t2)[i] : i < thrust::get<1>(t1) ? thrust::get<1>(t2)[i] : thrust::get<2>(t2)[i];
        }
    };
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
    join_iterator(const SizesTuple& t1, const Tuple& t2) : t1(t1), t2(t2) {}

    /*! \brief This method returns an iterator pointing to the beginning of
     *  this joined sequence of permuted entries.
     *  \return mStart
     */
    iterator begin(void) const
    {
        return TransformIterator(thrust::get<tuple_size-1>(t2), join_select_functor(t1,t2));
    }

    /*! \brief This method returns an iterator pointing to one element past
     *  the last of this joined sequence of permuted entries.
     *  \return mEnd
     */
    iterator end(void) const
    {
        return begin() + thrust::get<tuple_size-2>(t1);
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
    const SizesTuple& t1;
    const Tuple& t2;
    /*! \cond */
};

template <typename T1, typename T2, typename T3>
typename join_iterator< thrust::tuple<T1,T2,T3> >::iterator
make_join_iterator(const size_t s1, const size_t s2, const T1& t1, const T2& t2, const T3& t3)
{
    typedef thrust::tuple<T1,T2,T3>  Tuple;
    return join_iterator<Tuple>(thrust::make_tuple(s1, s1+s2), thrust::make_tuple(t1, t2-s1, t3)).begin();
}

template <typename T1, typename T2, typename T3, typename T4>
typename join_iterator< thrust::tuple<T1,T2,T3,T4> >::iterator
make_join_iterator(const size_t s1, const size_t s2, const size_t s3,
                   const T1& t1, const T2& t2, const T3& t3, const T4& t4)
{
    typedef thrust::tuple<T1,T2,T3,T4>  Tuple;
    return join_iterator<Tuple>(thrust::make_tuple(s1, s1+s2, s1+s2+s3), thrust::make_tuple(t1, t2-s1, t3-s1-s2, t4)).begin();
}

/*! \} // end iterators
 */

} // end namespace cusp

