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
 *  \brief Defines templated convenience functors analogous to what
 *         is found in thrust's functional.
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

/**
 * \p plus_value is a function object. Specifically, it is an Adaptable Unary Function.
 *  If \c f(c) is an object of class <tt>plus_value<T></tt>, and \c x is an object
 *  of class \c T, then <tt>f(x)</tt> returns <tt>x+c</tt>.
 *
 *  \param T is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and if \c x is an object of type \p T, then <tt>x+c</tt> must be defined
 *          and must have a return type that is convertible to \c T.
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/functional.h>
 * #include <cusp/print.h>
 *
 * #include <thrust/transform.h>
 *
 * int main()
 * {
 *    // create an array with 5 entries all set to 1
 *    cusp::constant_array<int> ones(5, 1);
 *
 *    // allocate size of transformed output array
 *    cusp::array1d<int,cusp::host_memory> v(5, 0);
 *
 *    // compute output vector as transform of ones
 *    thrust::transform(ones.begin(), ones.end(), v.begin(), cusp::plus_value<int>(2));
 *
 *    // v = [3, 3, 3, 3, 3]
 *    cusp::print(v);
 * }
 * \endcode
 */
template <typename T>
struct plus_value : public detail::base_functor< thrust::plus<T> >
{
    __host__ __device__
    plus_value(const T value = T(0)) : detail::base_functor< thrust::plus<T> >(value) {}
};

/**
 * \p divide_value is a function object. Specifically, it is an Adaptable Unary Function.
 *  If \c f(c) is an object of class <tt>divide_value<T></tt>, and \c x is an object
 *  of class \c T, then <tt>f(x)</tt> returns <tt>x/c</tt>.
 *
 *  \param T is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and if \c x is an object of type \p T, then <tt>x/c</tt> must be defined
 *          and must have a return type that is convertible to \c T.
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/functional.h>
 * #include <cusp/print.h>
 *
 * int main()
 * {
 *    // create an array with 5 entries all set to 10
 *    cusp::constant_array<int> tens(5, 10);
 *
 *    // allocate size of transformed output array
 *    cusp::array1d<int,cusp::host_memory> v(5, 0);
 *
 *    // compute output vector as transform of tens
 *    thrust::transform(tens.begin(), tens.end(), v.begin(), cusp::divide_value<int>(2));
 *
 *    // v = [5, 5, 5, 5, 5]
 *    cusp::print(v);
 * }
 * \endcode
 */
template <typename T>
struct divide_value : public detail::base_functor< thrust::divides<T> >
{
    __host__ __device__
    divide_value(const T value = T(0)) : detail::base_functor< thrust::divides<T> >(value) {}
};

/**
 * \p modulus_value is a function object. Specifically, it is an Adaptable Unary Function.
 *  If \c f(c) is an object of class <tt>modulus_value<T></tt>, and \c x is an object
 *  of class \c T, then <tt>f(x)</tt> returns <tt>x%c</tt>.
 *
 *  \param T is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and if \c x is an object of type \p T, then <tt>x%c</tt> must be defined
 *          and must have a return type that is convertible to \c T.
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/functional.h>
 * #include <cusp/print.h>
 *
 * int main()
 * {
 *    // create an array with 5 entries all set to 10
 *    cusp::constant_array<int> tens(5, 10);
 *
 *    // allocate size of transformed output array
 *    cusp::array1d<int,cusp::host_memory> v(5, 0);
 *
 *    // compute output vector as transform of tens
 *    thrust::transform(tens.begin(), tens.end(), v.begin(), cusp::modulus_value<int>(3));
 *
 *    // v = [1, 1, 1, 1, 1]
 *    cusp::print(v);
 * }
 * \endcode
 */
template <typename T>
struct modulus_value : public detail::base_functor< thrust::modulus<T> >
{
    __host__ __device__
    modulus_value(const T value = T(0)) : detail::base_functor< thrust::modulus<T> >(value) {}
};

/**
 * \p multiplies_value is a function object. Specifically, it is an Adaptable Unary Function.
 *  If \c f(c) is an object of class <tt>multiplies_value<T></tt>, and \c x is an object
 *  of class \c T, then <tt>f(x)</tt> returns <tt>x*c</tt>.
 *
 *  \param T is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and if \c x is an object of type \p T, then <tt>x*c</tt> must be defined
 *          and must have a return type that is convertible to \c T.
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/functional.h>
 * #include <cusp/print.h>
 *
 * int main()
 * {
 *    // create an array with 5 entries all set to 10
 *    cusp::constant_array<int> tens(5, 10);
 *
 *    // allocate size of transformed output array
 *    cusp::array1d<int,cusp::host_memory> v(5, 0);
 *
 *    // compute output vector as transform of tens
 *    thrust::transform(tens.begin(), tens.end(), v.begin(), cusp::multiplies_value<int>(3));
 *
 *    // v = [30, 30, 30, 30, 30]
 *    cusp::print(v);
 * }
 * \endcode
 */
template <typename T>
struct multiplies_value : public detail::base_functor< thrust::multiplies<T> >
{
    __host__ __device__
    multiplies_value(const T value) : detail::base_functor< thrust::multiplies<T> >(value) {}
};

/**
 * \p greater_value is a function object. Specifically, it is an Adaptable Unary Function.
 *  If \c f(c) is an object of class <tt>greater_value<T></tt>, and \c x is an object
 *  of class \c T, then <tt>f(x)</tt> returns <tt>x>c</tt>.
 *
 *  \param T is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and if \c x is an object of type \p T, then <tt>x>c</tt> must be defined
 *          and must have a return type that is convertible to \c T.
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/functional.h>
 * #include <cusp/print.h>
 *
 * int main()
 * {
 *    // create an array with 5 entries from 0 to 4
 *    cusp::counting_array<int> count(5);
 *
 *    // allocate size of transformed output array
 *    cusp::array1d<bool,cusp::host_memory> v(5, 0);
 *
 *    // compute output vector as transform of tens
 *    thrust::transform(count.begin(), count.end(), v.begin(), cusp::greater_value<int>(3));
 *
 *    // v = [0, 0, 0, 0, 1]
 *    cusp::print(v);
 * }
 * \endcode
 */
template <typename T>
struct greater_value : public detail::base_functor< thrust::greater<T> >
{
    __host__ __device__
    greater_value(const T value) : detail::base_functor< thrust::greater<T> >(value) {}
};

/**
 * \p greater_equal_value is a function object. Specifically, it is an Adaptable Unary Function.
 *  If \c f(c) is an object of class <tt>greater_equal_value<T></tt>, and \c x is an object
 *  of class \c T, then <tt>f(x)</tt> returns <tt>x>=c</tt>.
 *
 *  \param T is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and if \c x is an object of type \p T, then <tt>x>=c</tt> must be defined
 *          and must have a return type that is convertible to \c T.
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/functional.h>
 * #include <cusp/print.h>
 *
 * int main()
 * {
 *    // create an array with 5 entries from 0 to 4
 *    cusp::counting_array<int> count(5);
 *
 *    // allocate size of transformed output array
 *    cusp::array1d<bool,cusp::host_memory> v(5, 0);
 *
 *    // compute output vector as transform of tens
 *    thrust::transform(count.begin(), count.end(), v.begin(), cusp::greater_equal_value<int>(3));
 *
 *    // v = [0, 0, 0, 1, 1]
 *    cusp::print(v);
 * }
 * \endcode
 */
template <typename T>
struct greater_equal_value : public detail::base_functor< thrust::greater_equal<T> >
{
    __host__ __device__
    greater_equal_value(const T value) : detail::base_functor< thrust::greater_equal<T> >(value) {}
};

/**
 * \p less_value is a function object. Specifically, it is an Adaptable Unary Function.
 *  If \c f(c) is an object of class <tt>less_value<T></tt>, and \c x is an object
 *  of class \c T, then <tt>f(x)</tt> returns <tt>x<c</tt>.
 *
 *  \param T is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and if \c x is an object of type \p T, then <tt>x<c</tt> must be defined
 *          and must have a return type that is convertible to \c T.
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/functional.h>
 * #include <cusp/print.h>
 *
 * int main()
 * {
 *    // create an array with 5 entries from 0 to 4
 *    cusp::counting_array<int> count(5);
 *
 *    // allocate size of transformed output array
 *    cusp::array1d<bool,cusp::host_memory> v(5, 0);
 *
 *    // compute output vector as transform of tens
 *    thrust::transform(count.begin(), count.end(), v.begin(), cusp::less_value<int>(3));
 *
 *    // v = [1, 1, 1, 0, 0]
 *    cusp::print(v);
 * }
 * \endcode
 */
template <typename T>
struct less_value : public detail::base_functor< thrust::less<T> >
{
    __host__ __device__
    less_value(const T value) : detail::base_functor< thrust::less<T> >(value) {}
};

/**
 * \p less_equal_value is a function object. Specifically, it is an Adaptable Unary Function.
 *  If \c f(c) is an object of class <tt>less_equal_value<T></tt>, and \c x is an object
 *  of class \c T, then <tt>f(x)</tt> returns <tt>x<c</tt>.
 *
 *  \param T is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and if \c x is an object of type \p T, then <tt>x<=c</tt> must be defined
 *          and must have a return type that is convertible to \c T.
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/functional.h>
 * #include <cusp/print.h>
 *
 * int main()
 * {
 *    // create an array with 5 entries from 0 to 4
 *    cusp::counting_array<int> count(5);
 *
 *    // allocate size of transformed output array
 *    cusp::array1d<bool,cusp::host_memory> v(5, 0);
 *
 *    // compute output vector as transform of tens
 *    thrust::transform(count.begin(), count.end(), v.begin(), cusp::less_equal_value<int>(3));
 *
 *    // v = [1, 1, 1, 1, 0]
 *    cusp::print(v);
 * }
 * \endcode
 */
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
struct abs_functor : public thrust::unary_function<T, typename cusp::norm_type<T>::type>
{
    __host__ __device__
    typename cusp::norm_type<T>::type
    operator()(const T& t) const {
        return cusp::abs(t);
    }
};

template<typename T>
struct abs_squared_functor : public thrust::unary_function<T, typename cusp::norm_type<T>::type>
{
    __host__ __device__
    typename cusp::norm_type<T>::type
    operator()(const T& t) const {
        return cusp::square_functor<typename cusp::norm_type<T>::type>()(cusp::abs(t));
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
struct norm_functor : public thrust::unary_function<T, typename cusp::norm_type<T>::type>
{
    __host__ __device__
    typename cusp::norm_type<T>::type
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

