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

#include <cusp/detail/config.h>

#include <thrust/iterator/iterator_traits.h>

namespace cusp
{
namespace blas
{

template <typename ForwardIterator1,
          typename ForwardIterator2,
          typename ScalarType>
void axpy(ForwardIterator1 first1,
          ForwardIterator1 last1,
          ForwardIterator2 first2,
          ScalarType alpha);

template <typename Array1,
          typename Array2,
          typename ScalarType>
void axpy(const Array1& x,
                Array2& y,
          ScalarType alpha);


template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename ScalarType>
void axpby(InputIterator1 first1,
           InputIterator1 last1,
           InputIterator2 first2,
           OutputIterator output,
           ScalarType alpha,
           ScalarType beta);

template <typename Array1,
          typename Array2,
          typename Array3,
          typename ScalarType>
void axpby(const Array1& x,
           const Array2& y,
                 Array3& output,
          ScalarType alpha,
          ScalarType beta);


template <typename InputIterator1,
          typename InputIterator2,
          typename InputIterator3,
          typename OutputIterator,
          typename ScalarType>
void axpbypcz(InputIterator1 first1,
              InputIterator1 last1,
              InputIterator2 first2,
              InputIterator3 first3,
              OutputIterator output,
              ScalarType alpha,
              ScalarType beta,
              ScalarType gamma);

template <typename Array1,
          typename Array2,
          typename Array3,
          typename ScalarType>
void axpbypcz(const Array1& x,
              const Array2& y,
              const Array2& z,
                    Array3& output,
              ScalarType alpha,
              ScalarType beta,
              ScalarType gamma);


template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename ScalarType>
void xmy(InputIterator1 first1,
         InputIterator1 last1,
         InputIterator2 first2,
         OutputIterator output);

template <typename Array1,
          typename Array2,
          typename Array3>
void xmy(const Array1& x,
         const Array2& y,
               Array3& output);


template <typename InputIterator,
          typename ForwardIterator>
void copy(InputIterator   first1,
          InputIterator   last1,
          ForwardIterator first2);

template <typename Array1,
          typename Array2>
void copy(const Array1& array1,
                Array2& array2);


template <typename InputIterator1,
          typename InputIterator2>
typename thrust::iterator_value<InputIterator1>::type
    dot(InputIterator1 first1,
        InputIterator1 last1,
        InputIterator2 first2);

template <typename Array1,
          typename Array2>
typename Array1::value_type
    dot(const Array1& x,
        const Array2& y);


template <typename InputIterator1,
          typename InputIterator2>
typename thrust::iterator_value<InputIterator1>::type
    dotc(InputIterator1 first1,
         InputIterator1 last1,
         InputIterator2 first2);

template <typename Array1,
          typename Array2>
typename Array1::value_type
    dotc(const Array1& x,
         const Array2& y);


template <typename ForwardIterator,
          typename ScalarType>
void fill(ForwardIterator first,
          ForwardIterator last,
          ScalarType alpha);

template <typename Array,
          typename ScalarType>
void fill(Array& array,
          ScalarType alpha);


template <typename InputIterator>
typename thrust::iterator_value<InputIterator>::type
    nrm2(InputIterator first,
         InputIterator last);

template <typename Array>
typename Array::value_type
    nrm2(const Array& array);


template <typename ForwardIterator,
          typename ScalarType>
void scal(ForwardIterator first,
          ForwardIterator last,
          ScalarType alpha);

template <typename Array,
          typename ScalarType>
void scal(Array& array,
          ScalarType alpha);


} // end namespace blas
} // end namespace cusp

#include <cusp/detail/blas.inl>

