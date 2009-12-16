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

#include <cusp/exception.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>

#include <thrust/iterator/transform_iterator.h>

namespace cusp
{
namespace blas
{

namespace detail
{
    template <typename T1, typename MemorySpace1,
              typename T2, typename MemorySpace2>
    void assert_same_dimensions(const cusp::array1d<T1, MemorySpace1>& array1,
                                const cusp::array1d<T2, MemorySpace2>& array2)
    {
        if(array1.size() != array2.size())
            throw cusp::invalid_input_exception("array dimensions do not match");
    }
    
    template <typename T1, typename MemorySpace1,
              typename T2, typename MemorySpace2,
              typename T3, typename MemorySpace3>
    void assert_same_dimensions(const cusp::array1d<T1, MemorySpace1>& array1,
                                const cusp::array1d<T2, MemorySpace2>& array2,
                                const cusp::array1d<T3, MemorySpace3>& array3)
    {
        assert_same_dimensions(array1, array2);
        assert_same_dimensions(array2, array3);
    }
    
    template <typename T1, typename MemorySpace1,
              typename T2, typename MemorySpace2,
              typename T3, typename MemorySpace3,
              typename T4, typename MemorySpace4>
    void assert_same_dimensions(const cusp::array1d<T1, MemorySpace1>& array1,
                                const cusp::array1d<T2, MemorySpace2>& array2,
                                const cusp::array1d<T3, MemorySpace3>& array3,
                                const cusp::array1d<T4, MemorySpace4>& array4)
    {
        assert_same_dimensions(array1, array2);
        assert_same_dimensions(array2, array3);
        assert_same_dimensions(array3, array4);
    }

    // square<T> computes the square of a number f(x) -> x*x
    template <typename T>
        struct square : public thrust::unary_function<T,T>
        {
            __host__ __device__
                T operator()(T x)
                { 
                    return x * x;
                }
        };
    
    // conjugate<T> computes the complex conjugate of a number f(a + b * i) -> a - b * i
    template <typename T>
        struct conjugate : public thrust::unary_function<T,T>
        {
            __host__ __device__
                T operator()(T x)
                { 
                    // TODO actually handle complex numbers
                    return x;
                }
        };
    
    template <typename T>
        struct SCAL : public thrust::unary_function<T,T>
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
        struct AXPY : public thrust::binary_function<T,T,T>
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
        struct AXPBY : public thrust::binary_function<T,T,T>
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
    
    template <typename T>
        struct AXPBYPCZ
        {
            T alpha;
            T beta;
            T gamma;

            AXPBYPCZ(T _alpha, T _beta, T _gamma)
                : alpha(_alpha), beta(_beta), gamma(_gamma) {}

            template <typename Tuple>
            __host__ __device__
                void operator()(Tuple t)
                { 
                    thrust::get<3>(t) = alpha * thrust::get<0>(t) +
                                        beta  * thrust::get<1>(t) +
                                        gamma * thrust::get<2>(t);
                }
        };

    template <typename T>
        struct XMY : public thrust::binary_function<T,T,T>
        {
            __host__ __device__
                T operator()(T x, T y)
                { 
                    return x * y;
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
    detail::assert_same_dimensions(x, y);
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
    detail::assert_same_dimensions(x, y, z);
    cusp::blas::axpby(x.begin(), x.end(), y.begin(), z.begin(), alpha, beta);
}

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
              ScalarType gamma)
{
    size_t N = last1 - first1;
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(first1, first2, first3, output)),
                     thrust::make_zip_iterator(thrust::make_tuple(first1, first2, first3, output)) + N,
                     detail::AXPBYPCZ<ScalarType>(alpha, beta, gamma));
}

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
              ScalarType gamma)
{
    detail::assert_same_dimensions(x, y, z, output);
    cusp::blas::axpbypcz(x.begin(), x.end(), y.begin(), z.begin(), output.begin(), alpha, beta, gamma);
}
    

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator>
void xmy(InputIterator1 first1,
         InputIterator1 last1,
         InputIterator2 first2,
         OutputIterator output)
{
    typedef typename thrust::iterator_value<OutputIterator>::type ScalarType;
    thrust::transform(first1, last1, first2, output, detail::XMY<ScalarType>());
}

template <typename Array1,
          typename Array2,
          typename Array3>
void xmy(const Array1& x,
         const Array2& y,
               Array3& output)
{
    detail::assert_same_dimensions(x, y, output);
    cusp::blas::xmy(x.begin(), x.end(), y.begin(), output.begin());
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
    detail::assert_same_dimensions(x, y);
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
    detail::assert_same_dimensions(x, y);
    return cusp::blas::dot(x.begin(), x.end(), y.begin());
}

// TODO properly harmonize heterogenous types
template <typename InputIterator1,
          typename InputIterator2>
typename thrust::iterator_value<InputIterator1>::type
    dotc(InputIterator1 first1,
         InputIterator1 last1,
         InputIterator2 first2)
{
    typedef typename thrust::iterator_value<InputIterator1>::type OutputType;
    return thrust::inner_product(thrust::make_transform_iterator(first1, detail::conjugate<OutputType>()),
                                 thrust::make_transform_iterator(last1,  detail::conjugate<OutputType>()),
                                 first2,
                                 OutputType(0));
}

// TODO properly harmonize heterogenous types
template <typename Array1,
          typename Array2>
typename Array1::value_type
    dotc(const Array1& x,
         const Array2& y)
{
    detail::assert_same_dimensions(x, y);
    return cusp::blas::dotc(x.begin(), x.end(), y.begin());
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
    nrm2(const Array& x)
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

