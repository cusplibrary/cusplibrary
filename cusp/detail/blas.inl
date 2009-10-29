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
                T operator()(T x)
                { 
                    return x * x;
                }
        };
    
    template <typename T>
        struct SCAL
        {
            T alpha;

            SCAL(T _alpha) : alpha(_alpha) {}

            __host__ __device__
                T operator()(T x)
                { 
                    return alpha * x;
                }
        };

    template <typename T>
        struct AXPY
        {
            T alpha;

            AXPY(T _alpha) : alpha(_alpha) {}

            __host__ __device__
                T operator()(T x, T y)
                {
                    return alpha * x + y;
                }
        };
    
    template <typename T>
        struct AXPBY
        {
            T alpha;
            T beta;

            AXPBY(T _alpha, T _beta) : alpha(_alpha), beta(_beta) {}

            __host__ __device__
                T operator()(T x, T y)
                { 
                    return alpha * x + beta * y;
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
    thrust::transform(first1, last1, first2, first2, detail::AXPY<ScalarType>(alpha));
}

template <typename Array1,
          typename Array2,
          typename ScalarType>
void axpy(const Array1& x,
                Array2& y,
          ScalarType alpha)
{
    cusp::blas::axpy(x.begin(), x.end(), y.begin(), alpha);
}


template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename ScalarType>
void axpby(InputIterator1 first1,
           InputIterator1 last1,
           InputIterator2 first2,
           OutputIterator output,
           ScalarType alpha,
           ScalarType beta)
{
    thrust::transform(first1, last1, first2, output, detail::AXPBY<ScalarType>(alpha, beta));
}

template <typename Array1,
          typename Array2,
          typename Array3,
          typename ScalarType>
void axpby(const Array1& x,
           const Array2& y,
                 Array3& z,
          ScalarType alpha,
          ScalarType beta)
{
    cusp::blas::axpby(x.begin(), x.end(), y.begin(), z.begin(), alpha, beta);
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
void copy(const Array1& x,
                Array2& y)
{
    cusp::blas::copy(x.begin(), x.end(), y.begin());
}


// TODO properly harmonize heterogenous types
template <typename InputIterator1,
          typename InputIterator2>
typename thrust::iterator_value<InputIterator1>::type
    dot(InputIterator1 first1,
        InputIterator1 last1,
        InputIterator2 first2)
{
    typedef typename thrust::iterator_value<InputIterator1>::type OutputType;
    return thrust::inner_product(first1, last1, first2, OutputType(0));
}

// TODO properly harmonize heterogenous types
template <typename Array1,
          typename Array2>
typename Array1::value_type
    dot(const Array1& x,
        const Array2& y)
{
    return cusp::blas::dot(x.begin(), x.end(), y.begin());
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
void fill(Array& x,
          ScalarType alpha)
{
    cusp::blas::fill(x.begin(), x.end(), alpha);
}


template <typename InputIterator>
typename thrust::iterator_value<InputIterator>::type
    nrm2(InputIterator first,
         InputIterator last)
{
    typedef typename thrust::iterator_value<InputIterator>::type ValueType;

    detail::square<ValueType> unary_op;
    thrust::plus<ValueType>   binary_op;

    ValueType init = 0;

    return std::sqrt( thrust::transform_reduce(first, last, unary_op, init, binary_op) );
}

template <typename Array>
typename Array::value_type
    nrm2(Array& x)
{
    return cusp::blas::nrm2(x.begin(), x.end());
}


template <typename ForwardIterator,
          typename ScalarType>
void scal(ForwardIterator first,
          ForwardIterator last,
          ScalarType alpha)
{
    thrust::transform(first, last, first, detail::SCAL<ScalarType>(alpha));
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

