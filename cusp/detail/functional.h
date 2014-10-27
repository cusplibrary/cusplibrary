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

