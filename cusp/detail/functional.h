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

template<typename T>
struct zero_function : public thrust::unary_function<T,T>
{
    __host__ __device__ T operator()(const T &x) const {
        return T(0);
    }
}; // end minus

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

template <typename T>
struct sqrt_functor : thrust::unary_function<T,T>
{
    __host__ __device__
    T operator()(const T& x) {
        return sqrt(x);
    }
};


// absolute<T> computes the absolute value of a number f(x) -> |x|
template <typename T>
struct absolute : public thrust::unary_function<T,T>
{
    __host__ __device__
    T operator()(T x)
    {
        return abs(x);
    }
};

template <typename T, typename BinaryFunction>
struct base_functor : public thrust::unary_function<T,T>
{
    const T value;
    BinaryFunction op;

    base_functor(const T value) : value(value) {}

    __host__ __device__
    T operator()(const T x)
    {
        return op(x, value);
    }
};

template <typename T>
struct modulus_value : public base_functor< T, thrust::modulus<T> >
{
    typedef base_functor< T,thrust::modulus<T> > Parent;
    modulus_value(const T value) : Parent(value) {}
};

template <typename T>
struct divide_value : public base_functor< T, thrust::divides<T> >
{
    typedef base_functor< T,thrust::divides<T> > Parent;
    divide_value(const T value) : Parent(value) {}
};

template <typename T>
struct greater_than_or_equal_to
{
    const T num;

    greater_than_or_equal_to(const T num)
        : num(num) {}

    __host__ __device__ bool operator()(const T &x) const {
        return x >= num;
    }
};

template <typename T>
struct less_than
{
    const T num;

    less_than(const T num)
        : num(num) {}

    __host__ __device__ bool operator()(const T &x) const {
        return x < num;
    }
};

template <typename IndexType>
struct is_positive
{
    __host__ __device__
    bool operator()(const IndexType x)
    {
        return x > 0;
    }
};

template <typename IndexType>
struct empty_row_functor
{
    typedef bool result_type;

    template <typename Tuple>
    __host__ __device__
    bool operator()(const Tuple& t) const
    {
        const IndexType a = thrust::get<0>(t);
        const IndexType b = thrust::get<1>(t);

        return a != b;
    }
};

template <typename IndexType>
struct diagonal_index_functor : public thrust::unary_function<IndexType,IndexType>
{
    const IndexType pitch;

    diagonal_index_functor(const IndexType pitch)
        : pitch(pitch) {}

    template <typename Tuple>
    __host__ __device__
    IndexType operator()(const Tuple& t) const
    {
        const IndexType row  = thrust::get<0>(t);
        const IndexType diag = thrust::get<1>(t);

        return (diag * pitch) + row;
    }
};

template <typename T, typename BinaryFunction>
struct combine_tuple_base_functor : public thrust::unary_function<T,T>
{
    BinaryFunction op;

    template<typename Tuple>
    __host__ __device__
    T operator()(const Tuple& t)
    {
        return op(thrust::get<0>(t),thrust::get<1>(t));
    }
};

template <typename T>
struct sum_tuple_functor : public thrust::unary_function<T,T>
{
    template<typename Tuple>
    __host__ __device__
    T operator()(const Tuple& t)
    {
        return thrust::get<0>(t) + thrust::get<1>(t);
    }
};

template <typename IndexType>
struct occupied_diagonal_functor
{
    typedef IndexType result_type;

    const   IndexType num_rows;

    occupied_diagonal_functor(const IndexType num_rows)
        : num_rows(num_rows) {}

    template <typename Tuple>
    __host__ __device__
    IndexType operator()(const Tuple& t) const
    {
        const IndexType i = thrust::get<0>(t);
        const IndexType j = thrust::get<1>(t);

        return j-i+num_rows;
    }
};

struct speed_threshold_functor
{
    size_t num_rows;
    float  relative_speed;
    size_t breakeven_threshold;

    speed_threshold_functor(const size_t num_rows, const float relative_speed, const size_t breakeven_threshold)
        : num_rows(num_rows),
          relative_speed(relative_speed),
          breakeven_threshold(breakeven_threshold)
    {}

    template <typename IndexType>
    __host__ __device__
    bool operator()(const IndexType rows) const
    {
        return relative_speed * (num_rows-rows) < num_rows || (size_t) (num_rows-rows) < breakeven_threshold;
    }
};

template<typename IndexType>
struct row_operator : public std::unary_function<size_t,IndexType>
{
    size_t pitch;

    row_operator(size_t pitch)
        : pitch(pitch) {}

    __host__ __device__
    IndexType operator()(const size_t & linear_index) const
    {
        return linear_index % pitch;
    }
};

template <typename IndexType>
struct is_valid_ell_index
{
    const IndexType num_rows;

    is_valid_ell_index(const IndexType num_rows)
        : num_rows(num_rows) {}

    template <typename Tuple>
    __host__ __device__
    bool operator()(const Tuple& t) const
    {
        const IndexType i = thrust::get<0>(t);
        const IndexType j = thrust::get<1>(t);

        return i < num_rows && j != IndexType(-1);
    }
};

template <typename IndexType, typename ValueType>
struct is_valid_coo_index
{
    const IndexType num_rows;
    const IndexType num_cols;

    is_valid_coo_index(const IndexType num_rows, const IndexType num_cols)
        : num_rows(num_rows), num_cols(num_cols) {}

    template <typename Tuple>
    __host__ __device__
    bool operator()(const Tuple& t) const
    {
        const IndexType i = thrust::get<0>(t);
        const IndexType j = thrust::get<1>(t);
        const ValueType value = thrust::get<2>(t);

        return ( i > IndexType(-1) && i < num_rows ) &&
               ( j > IndexType(-1) && j < num_cols ) &&
               ( value != ValueType(0) ) ;
    }
};

template <typename IndexType>
struct tuple_equal_to : public thrust::unary_function<thrust::tuple<IndexType,IndexType>,bool>
{
    __host__ __device__
    bool operator()(const thrust::tuple<IndexType,IndexType>& t) const
    {
        return thrust::get<0>(t) == thrust::get<1>(t);
    }
};


// maximum<T> returns the largest of two numbers
template <typename T>
struct maximum : public thrust::binary_function<T,T,T>
{
    __host__ __device__
    T operator()(T x, T y)
    {
        return thrust::maximum<T>()(x,y);
    }
};

// maximum<T> returns the number with the largest real part
template <typename T>
struct maximum< cusp::complex<T> > : public thrust::binary_function<cusp::complex<T>,cusp::complex<T>,cusp::complex<T> >
{
    __host__ __device__
    cusp::complex<T> operator()(cusp::complex<T> x, cusp::complex<T> y)
    {
        return thrust::maximum<T>()(x.real(),y.real());
    }
};

// conjugate<T> computes the complex conjugate of a number f(a + b * i) -> a - b * i
template <typename T>
struct conjugate : public thrust::unary_function<T,T>
{
    __host__ __device__
    T operator()(T x)
    {
        return x;
    }
};

template <typename T>
struct conjugate<cusp::complex<T> > : public thrust::unary_function<cusp::complex<T>,
        cusp::complex<T> >
{
    __host__ __device__
    cusp::complex<T> operator()(cusp::complex<T> x)
    {
        return thrust::conj(x);
    }
};

// square<T> computes the square of a number f(x) -> x*conj(x)
template <typename T>
struct norm_squared : public thrust::unary_function<T,T>
{
    __host__ __device__
    T operator()(T x)
    {
        return x * conjugate<T>()(x);
    }
};
template <typename T>
struct SCAL
{
    T alpha;

    SCAL(T _alpha)
        : alpha(_alpha) {}

    template <typename T2>
    __host__ __device__
    void operator()(T2 & x)
    {
        x = alpha * x;
    }
};


template <typename T>
struct AXPY
{
    T alpha;

    AXPY(T _alpha)
        : alpha(_alpha) {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        thrust::get<1>(t) = alpha * thrust::get<0>(t) +
                            thrust::get<1>(t);
    }
};

template <typename T1, typename T2>
struct AXPBY
{
    T1 alpha;
    T2 beta;

    AXPBY(T1 _alpha, T2 _beta)
        : alpha(_alpha), beta(_beta) {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        thrust::get<2>(t) = alpha * thrust::get<0>(t) +
                            beta  * thrust::get<1>(t);
    }
};

template <typename T1,typename T2,typename T3>
struct AXPBYPCZ
{
    T1 alpha;
    T2 beta;
    T3 gamma;

    AXPBYPCZ(T1 _alpha, T2 _beta, T3 _gamma)
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
} // end namespace cusp

