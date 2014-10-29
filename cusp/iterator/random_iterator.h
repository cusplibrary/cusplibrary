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

/*! \cond */
namespace detail
{
  // Forward definition
  template<typename,typename> struct random_integer_functor;
} // end detail
/*! \endcond */

/*! \addtogroup iterators Iterators
 *  \ingroup iterators
 *  \brief Various customized Thrust based iterators
 *  \{
 */

/**
 *  \brief
 *
 *  \par Overview
 *  \p random_iterator is an iterator which represents a pointer into a range
 *  of constant values. This iterator is useful for creating a range filled with the same
 *  value without explicitly storing it in memory. Using \p random_iterator saves both
 *  memory capacity and bandwidth.
 *
 *  \par Example
 *  The following code snippet demonstrates how to create a \p random_iterator whose
 *  \c value_type is \c int and whose seed is \c 5.
 *
 *  \code
 *  #include <cusp/iterator/random_iterator.h>
 *
 *  int main()
 *  {
 *    cusp::random_iterator<int> iter(5);
 *
 *    std::cout << iter[0] << std::endl;
 *    std::cout << iter[1] << std::endl;
 *    std::cout << iter[2] << std::endl;
 *  }
 *  \endcode
 */
template<typename T>
class random_iterator
{
public:

    /*! \cond */
    typedef T                                                                            value_type;
    typedef T*                                                                           pointer;
    typedef T&                                                                           reference;
    typedef size_t                                                                       difference_type;
    typedef size_t                                                                       size_type;
    typedef thrust::random_access_traversal_tag                                          iterator_category;

    typedef std::ptrdiff_t                                                               IndexType;
    typedef detail::random_integer_functor<IndexType,T>                                  IndexFunctor;
    typedef typename thrust::counting_iterator<IndexType>                                RandomCountingIterator;
    typedef typename thrust::transform_iterator<IndexFunctor, RandomCountingIterator, T> RandomTransformIterator;

    // type of the random_range iterator
    typedef RandomTransformIterator                                                      iterator;
    /*! \endcond */

    IndexFunctor index_func;

    random_iterator(size_t seed)
        : index_func(seed) {}

    iterator begin(void) const
    {
        return RandomTransformIterator(RandomTransformIterator(RandomCountingIterator(0), index_func), index_func);
    }
}; // end random_iterator

/*! \} // end iterators
 */

} // end namespace cusp

