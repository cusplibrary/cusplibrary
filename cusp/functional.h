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


/*! \file functional.h
 *  \brief Defines templated functors and traits analogous to what
 *         is found in stl and boost's functional.
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/complex.h>

#include <thrust/functional.h>

namespace cusp
{
namespace detail
{
template <typename> struct base_functor;
template <typename> struct combine_tuple_base_functor;
}

template <typename T>
struct plus_value : public detail::base_functor< thrust::plus<T> >
{
    __host__ __device__
    plus_value(const T value = T(0)) : detail::base_functor< thrust::plus<T> >(value) {}
};

template <typename T>
struct divide_value : public detail::base_functor< thrust::divides<T> >
{
    __host__ __device__
    divide_value(const T value = T(0)) : detail::base_functor< thrust::divides<T> >(value) {}
};

template <typename T>
struct modulus_value : public detail::base_functor< thrust::modulus<T> >
{
    __host__ __device__
    modulus_value(const T value = T(0)) : detail::base_functor< thrust::modulus<T> >(value) {}
};

template <typename T>
struct multiplies_value : public detail::base_functor< thrust::multiplies<T> >
{
    __host__ __device__
    multiplies_value(const T value) : detail::base_functor< thrust::multiplies<T> >(value) {}
};

template <typename T>
struct greater_value : public detail::base_functor< thrust::greater<T> >
{
    __host__ __device__
    greater_value(const T value) : detail::base_functor< thrust::greater<T> >(value) {}
};

template <typename T>
struct greater_equal_value : public detail::base_functor< thrust::greater_equal<T> >
{
    __host__ __device__
    greater_equal_value(const T value) : detail::base_functor< thrust::greater_equal<T> >(value) {}
};

template <typename T>
struct less_value : public detail::base_functor< thrust::less<T> >
{
    __host__ __device__
    less_value(const T value) : detail::base_functor< thrust::less<T> >(value) {}
};

template <typename T>
struct less_equal_value : public detail::base_functor< thrust::less_equal<T> >
{
    __host__ __device__
    less_equal_value(const T value) : detail::base_functor< thrust::less_equal<T> >(value) {}
};

template<typename T>
struct zero_functor : public thrust::unary_function<T,T>
{
    __host__ __device__
    T operator()(const T& x) const {
        return T(0);
    }
};

// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct square_functor : public thrust::unary_function<T,T>
{
    __host__ __device__
    T operator()(const T& x) const {
        return x * x;
    }
};

template <typename T>
struct sqrt_functor : public thrust::unary_function<T,T>
{
    __host__ __device__
    T operator()(const T& x) const {
        using thrust::sqrt;
        using std::sqrt;

        return sqrt(x);
    }
};

template <typename T>
struct reciprocal : public thrust::unary_function<T,T>
{
    __host__ __device__
    T operator()(const T& v) const {
        return T(1.0) / v;
    }
};

template<typename T>
struct abs_functor : public thrust::unary_function<T, typename cusp::detail::norm_type<T>::type>
{
    __host__ __device__
    typename cusp::detail::norm_type<T>::type
    operator()(const T& t) const {
        return cusp::abs(t);
    }
};

template<typename T>
struct abs_squared_functor : public thrust::unary_function<T, typename cusp::detail::norm_type<T>::type>
{
    __host__ __device__
    typename cusp::detail::norm_type<T>::type
    operator()(const T& t) const {
        return cusp::square_functor()(cusp::abs(t));
    }
};

template<typename T>
struct conj_functor : public thrust::unary_function<T,T>
{
    __host__ __device__
    T operator()(const T& t) const {
        return cusp::conj(t);
    }
};

template<typename T>
struct norm_functor : public thrust::unary_function<T, typename cusp::detail::norm_type<T>::type>
{
    __host__ __device__
    const typename cusp::detail::norm_type<T>::type
    operator()(const T& t) const {
        return cusp::norm(t);
    }
};

template <typename T>
struct sum_tuple_functor : public detail::combine_tuple_base_functor< thrust::plus<T> > {};

template <typename T>
struct divide_tuple_functor : public detail::combine_tuple_base_functor< thrust::divides<T> > {};

template <typename T>
struct equal_tuple_functor : public detail::combine_tuple_base_functor< thrust::equal_to<T> > {};

template <typename T>
struct not_equal_tuple_functor : public detail::combine_tuple_base_functor< thrust::not_equal_to<T> > {};

} // end namespace cusp

#include <cusp/detail/functional.inl>

