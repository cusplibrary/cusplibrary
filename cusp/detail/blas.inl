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

#include <cusp/array1d.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>

namespace cusp
{
namespace blas
{

namespace detail
{
    // square<T> computes the square of a number f(x) -> x*x
    template <typename T>
        struct square
        {
            __host__ __device__
                T operator()(const T& x) const { 
                    return x * x;
                }
        };
    
    template <typename T>
        struct scale
        {
            const T a;

            scale(const T& _a) : a(_a) {}

            __host__ __device__
                T operator()(const T& x) const { 
                    return a * x;
                }
        };

    template <typename T>
        struct scale_and_add
        {
            const T a;

            scale_and_add(const T& _a) : a(_a) {}

            __host__ __device__
                T operator()(const T& x, const T& y) const { 
                    return a * x + y;
                }
        };

} // end namespace detail


template <typename ForwardIterator1,
          typename ForwardIterator2,
          typename ScalarType>
void axpy(ForwardIterator1 first1,
          ForwardIterator1 last1,
          ForwardIterator2 first2,
          ScalarType alpha)
{
    thrust::transform(first1, last1, first2, first2, detail::scale_and_add<ScalarType>(alpha));
}

template <typename Array,
          typename ScalarType>
void axpy(const Array& array1,
                Array& array2,
          ScalarType alpha)
{
    cusp::blas::axpy(array1.begin(), array1.end(), array2.begin(), alpha);
}


template <typename InputIterator,
          typename ForwardIterator>
void copy(InputIterator   first1,
          InputIterator   last1,
          ForwardIterator first2)
{
    thrust::copy(first1, last1, first2);
}

template <typename Array1,
          typename Array2>
void copy(const Array1& array1,
                Array2& array2)
{
    cusp::blas::copy(array1.begin(), array1.end(), array2.begin());
}



template <typename ForwardIterator>
typename thrust::iterator_value<ForwardIterator>::type
    dot(ForwardIterator first1,
        ForwardIterator last1,
        ForwardIterator first2)
{
    typedef typename thrust::iterator_value<ForwardIterator>::type OutputType;
    return thrust::inner_product(first1, last1, first2, OutputType(0));
}

template <typename Array>
typename Array::value_type
    dot(const Array& array1,
        const Array& array2)
{
    return cusp::blas::dot(array1.begin(), array1.end(), array2.begin());
}


template <typename ForwardIterator,
          typename ScalarType>
void fill(ForwardIterator first,
          ForwardIterator last,
          ScalarType alpha)
{
    thrust::fill(first, last, alpha);
}

template <typename Array,
          typename ScalarType>
void fill(Array& array,
          ScalarType alpha)
{
    cusp::blas::fill(array.begin(), array.end(), alpha);
}


template <typename InputIterator>
typename thrust::iterator_value<InputIterator>::type
    nrm2(InputIterator first,
         InputIterator last)
{
    typedef typename thrust::iterator_value<InputIterator>::type ValueType;

    detail::square<ValueType> unary_op;
    thrust::plus<ValueType>  binary_op;

    ValueType init = 0;

    return std::sqrt( thrust::transform_reduce(first, last, unary_op, init, binary_op) );
}

template <typename Array>
typename Array::value_type
    nrm2(Array& array)
{
    return cusp::blas::nrm2(array.begin(), array.end());
}


template <typename ForwardIterator,
          typename ScalarType>
void scal(ForwardIterator first,
          ForwardIterator last,
          ScalarType alpha)
{
    thrust::transform(first, last, first, detail::scale<ScalarType>(alpha));
}

template <typename Array,
          typename ScalarType>
void scal(Array& array,
          ScalarType alpha)
{
    cusp::blas::scal(array.begin(), array.end(), alpha);
}

} // end namespace blas
} // end namespace cusp

