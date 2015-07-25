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

#include <cusp/array1d.h>
#include <cusp/exception.h>
#include <cusp/functional.h>
#include <cusp/print.h>

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
namespace thrustblas
{
namespace detail
{

template <typename T>
struct SCAL
{
    T alpha;

    SCAL(T _alpha)
        : alpha(_alpha) {}

    template <typename T2>
    __host__ __device__
    void operator()(T2& x)
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
    void operator()(Tuple& t)
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
    void operator()(Tuple& t)
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
    void operator()(Tuple& t)
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
    T operator()(const T& x, const T& y)
    {
        return x * y;
    }
};

} // end detail thrustblas

template <typename DerivedPolicy,
          typename Array>
int amax(thrust::execution_policy<DerivedPolicy>& exec,
         const Array& x)
{
    typedef typename Array::value_type ValueType;

    cusp::abs_functor<ValueType> unary_op;

    return thrust::max_element(exec,
                               thrust::make_transform_iterator(x.begin(), unary_op),
                               thrust::make_transform_iterator(x.end(), unary_op))
           - thrust::make_transform_iterator(x.begin(), unary_op);
}

template <typename DerivedPolicy,
          typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
asum(thrust::execution_policy<DerivedPolicy>& exec,
     const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename Array::memory_space MemorySpace;
    typedef typename cusp::detail::norm_type<ValueType>::type NormType;

    cusp::abs_functor<ValueType> unary_op;
    thrust::plus<NormType>       binary_op;

    NormType init = 0;

    return thrust::transform_reduce(exec, x.begin(), x.end(), unary_op, init, binary_op);
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2,
          typename ScalarType>
void axpy(thrust::execution_policy<DerivedPolicy>& exec,
          const Array1& x,
                Array2& y,
          const ScalarType alpha)
{
    typedef typename Array1::value_type ValueType;

    size_t N = x.size();

    thrust::for_each(exec,
                     thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin())) + N,
                     detail::AXPY<ValueType>(alpha));
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2,
          typename Array3,
          typename ScalarType1,
          typename ScalarType2>
void axpby(thrust::execution_policy<DerivedPolicy> &exec,
           const Array1& x,
           const Array2& y,
                 Array3& z,
                 ScalarType1 alpha,
                 ScalarType2 beta)
{
    typedef typename Array1::value_type ValueType;

    size_t N = x.size();

    thrust::for_each(exec,
                     thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin())) + N,
                     detail::AXPBY<ValueType,ValueType>(alpha, beta));
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2,
          typename Array3,
          typename Array4,
          typename ScalarType1,
          typename ScalarType2,
          typename ScalarType3>
void axpbypcz(thrust::execution_policy<DerivedPolicy> &exec,
              const Array1& x,
              const Array2& y,
              const Array3& z,
                    Array4& output,
                    ScalarType1 alpha,
                    ScalarType2 beta,
                    ScalarType3 gamma)
{
    typedef typename Array1::value_type ValueType;

    size_t N = x.size();

    thrust::for_each(exec,
                     thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin(), output.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), z.begin(), output.begin())) + N,
                     detail::AXPBYPCZ<ValueType,ValueType,ValueType>(alpha, beta, gamma));
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2,
          typename Array3>
void xmy(thrust::execution_policy<DerivedPolicy> &exec,
         const Array1& x,
         const Array2& y,
               Array3& z)
{
    typedef typename Array3::value_type ValueType;

    thrust::transform(exec, x.begin(), x.end(), y.begin(), z.begin(), detail::XMY<ValueType>());
}

template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
void copy(thrust::execution_policy<DerivedPolicy>& exec,
          const Array1& x,
          Array2& y)
{
    thrust::copy(exec, x.begin(), x.end(), y.begin());
}

// TODO properly harmonize heterogenous types
template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
typename Array1::value_type
dot(thrust::execution_policy<DerivedPolicy>& exec,
    const Array1& x,
    const Array2& y)
{
    typedef typename Array1::value_type OutputType;

    return thrust::inner_product(exec, x.begin(), x.end(), y.begin(), OutputType(0));
}

// TODO properly harmonize heterogenous types
template <typename DerivedPolicy,
          typename Array1,
          typename Array2>
typename Array1::value_type
dotc(thrust::execution_policy<DerivedPolicy>& exec,
     const Array1& x,
     const Array2& y)
{
    typedef typename Array1::value_type OutputType;

    return thrust::inner_product(exec,
                                 thrust::make_transform_iterator(x.begin(), cusp::conj_functor<OutputType>()),
                                 thrust::make_transform_iterator(x.end(),   cusp::conj_functor<OutputType>()),
                                 y.begin(),
                                 OutputType(0));
}

template <typename DerivedPolicy,
          typename Array1,
          typename ScalarType>
void fill(thrust::execution_policy<DerivedPolicy>& exec,
          Array1& x,
          const ScalarType alpha)
{
    thrust::fill(exec, x.begin(), x.end(), alpha);
}

template <typename DerivedPolicy,
          typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm2(thrust::execution_policy<DerivedPolicy>& exec,
     const Array& x)
{
    typedef typename Array::value_type ValueType;
    typedef typename Array::memory_space MemorySpace;
    typedef typename cusp::detail::norm_type<ValueType>::type NormType;

    cusp::abs_squared_functor<ValueType> unary_op;
    thrust::plus<NormType>               binary_op;

    NormType init = 0;

    return std::sqrt(thrust::transform_reduce(exec, x.begin(), x.end(), unary_op, init, binary_op));
}

template <typename DerivedPolicy,
          typename Array,
          typename ScalarType>
void scal(thrust::execution_policy<DerivedPolicy>& exec,
          Array& x,
          const ScalarType alpha)
{
    thrust::for_each(exec, x.begin(), x.end(), detail::SCAL<ScalarType>(alpha));
}

template<typename DerivedPolicy,
         typename Array2d,
         typename Array1d1,
         typename Array1d2>
void gemv(thrust::execution_policy<DerivedPolicy>& exec,
          const Array2d&  A,
          const Array1d1& x,
                Array1d2& y)
{
    throw cusp::not_implemented_exception("CUSP GEMV not implemented");
}

template <typename DerivedPolicy,
          typename Array1d1,
          typename Array1d2,
          typename Array2d1>
void ger(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
         const Array1d1& x,
         const Array1d2& y,
               Array2d1& A)
{
    throw cusp::not_implemented_exception("CUSP GER not implemented");
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array1d1,
          typename Array1d2>
void symv(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d1& A,
          const Array1d1& x,
                Array1d2& y)
{
    throw cusp::not_implemented_exception("CUSP SYMV not implemented");
}

template <typename DerivedPolicy,
          typename Array1d,
          typename Array2d>
void syr(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
         const Array1d& x,
               Array2d& A)
{
    throw cusp::not_implemented_exception("CUSP SYR not implemented");
}

template <typename DerivedPolicy,
          typename Array2d,
          typename Array1d>
void trmv(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d& A,
                Array1d& x)
{
    throw cusp::not_implemented_exception("CUSP TRMV not implemented");
}

template <typename DerivedPolicy,
          typename Array2d,
          typename Array1d>
void trsv(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d& A,
                Array1d& x)
{
    throw cusp::not_implemented_exception("CUSP TRSV not implemented");
}

template<typename DerivedPolicy,
         typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void gemm(thrust::execution_policy<DerivedPolicy>& exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C)
{
    throw cusp::not_implemented_exception("CUSP GEMM not implemented");
}

template<typename DerivedPolicy,
         typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void symm(thrust::execution_policy<DerivedPolicy>& exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C)
{
    throw cusp::not_implemented_exception("CUSP SYMM not implemented");
}

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2>
void syrk(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d1& A,
                Array2d2& B)
{
    throw cusp::not_implemented_exception("CUSP SYRK not implemented");
}

template<typename DerivedPolicy,
         typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void syr2k(thrust::execution_policy<DerivedPolicy>& exec,
           const Array2d1& A,
           const Array2d2& B,
                 Array2d3& C)
{
    throw cusp::not_implemented_exception("CUSP SYR2K not implemented");
}

template<typename DerivedPolicy,
         typename Array2d1,
         typename Array2d2>
void trmm(thrust::execution_policy<DerivedPolicy>& exec,
          const Array2d1& A,
                Array2d2& B)
{
    throw cusp::not_implemented_exception("CUSP TRMM not implemented");
}

template<typename DerivedPolicy,
         typename Array2d1,
         typename Array2d2>
void trsm(thrust::execution_policy<DerivedPolicy>& exec,
          const Array2d1& A,
                Array2d2& B)
{
    throw cusp::not_implemented_exception("CUSP TRSM not implemented");
}

template <typename DerivedPolicy,
          typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm1(thrust::execution_policy<DerivedPolicy>& exec,
     const Array& x)
{
    return cusp::blas::asum(exec, x);
}

template <typename DerivedPolicy,
          typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrmmax(thrust::execution_policy<DerivedPolicy>& exec,
       const Array& x)
{
    typedef typename Array::value_type ValueType;

    int index = cusp::blas::amax(exec, x);
    ValueType val = x[index];

    return cusp::abs(val);
}

} // end namespace thrustblas
} // end namespace blas
} // end namespace cusp

