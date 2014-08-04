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

#include <cmath>

namespace cusp
{
namespace blas
{
namespace detail
{
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
struct maximum<cusp::complex<T> > : public thrust::binary_function<cusp::complex<T>,cusp::complex<T>,cusp::complex<T> >
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
        return cusp::conj(x);
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

template <typename Array1,
         typename Array2,
         typename ScalarType>
void axpy(const thrustblas::detail::blas_policy<typename Array2::memory_space>& policy,
          const Array1& x,
          Array2& y,
          ScalarType alpha)
{
    size_t N = x.size();
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin())) + N,
                     detail::AXPY<ScalarType>(alpha));
}

// TODO properly harmonize heterogenous types
template <typename Array1,
         typename Array2>
typename Array1::value_type
dot(const thrustblas::detail::blas_policy<typename Array1::memory_space>& policy,
    const Array1& x,
    const Array2& y)
{
    typedef typename Array1::value_type OutputType;

    return thrust::inner_product(x.begin(), x.end(), y.begin(), OutputType(0));
}

// TODO properly harmonize heterogenous types
template <typename Array1,
         typename Array2>
typename Array1::value_type
dotc(const thrustblas::detail::blas_policy<typename Array1::memory_space>& policy,
     const Array1& x,
     const Array2& y)
{
    typedef typename Array1::value_type OutputType;

    return thrust::inner_product(thrust::make_transform_iterator(x.begin(), detail::conjugate<OutputType>()),
                                 thrust::make_transform_iterator(x.end(),  detail::conjugate<OutputType>()),
                                 y.begin(),
                                 OutputType(0));
}

template <typename Array>
typename norm_type<typename Array::value_type>::type
nrm1(const thrustblas::detail::blas_policy<typename Array::memory_space>& policy, Array& x)
{
    typedef typename Array::value_type ValueType;

    detail::absolute<ValueType> unary_op;
    thrust::plus<ValueType>     binary_op;

    ValueType init = 0;

    return abs(thrust::transform_reduce(x.begin(), x.end(), unary_op, init, binary_op));
}

template <typename Array>
typename norm_type<typename Array::value_type>::type
nrm2(const thrustblas::detail::blas_policy<typename Array::memory_space>& policy, Array& x)
{
    typedef typename Array::value_type ValueType;

    detail::norm_squared<ValueType> unary_op;
    thrust::plus<ValueType>   binary_op;

    ValueType init = 0;

    return std::sqrt( abs(thrust::transform_reduce(x.begin(), x.end(), unary_op, init, binary_op)) );
}

template <typename Array>
typename Array::value_type
nrmmax(const thrustblas::detail::blas_policy<typename Array::memory_space>& policy, Array& x)
{
    typedef typename Array::value_type ValueType;

    detail::absolute<ValueType>  unary_op;
    detail::maximum<ValueType>   binary_op;

    ValueType init = 0;

    return thrust::transform_reduce(x.begin(), x.end(), unary_op, init, binary_op);
}

template <typename Array,
         typename ScalarType>
void scal(const thrustblas::detail::blas_policy<typename Array::memory_space>& policy,
          Array& x, ScalarType alpha)
{
    thrust::for_each(x.begin(),
                     x.end(),
                     detail::SCAL<ScalarType>(alpha));
}

template<typename Array2d,
         typename Array1,
         typename Array2>
void gemv(const thrustblas::detail::blas_policy<typename Array2::memory_space>& policy,
          const Array2d& A,
          const Array1& x,
          Array2& y)
{
}

template<typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void gemm(const thrustblas::detail::blas_policy<typename Array2d3::memory_space>& policy,
          const Array2d1& A,
          const Array2d2& B,
          Array2d3& C)
{
}

} // end namespace blas
} // end namespace cusp

