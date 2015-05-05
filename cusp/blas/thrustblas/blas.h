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
#include <cusp/detail/functional.h>

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

template <typename DerivedPolicy,
          typename Array>
int amax(thrust::execution_policy<DerivedPolicy>& exec,
         const Array& x)
{
    typedef typename Array::value_type ValueType;

    detail::absolute<ValueType> unary_op;

    return thrust::max_element(exec,
                               thrust::make_transform_iterator(x.begin(), unary_op),
                               thrust::make_transform_iterator(x.end(), unary_op))
           - x.begin();
}

template <typename DerivedPolicy,
          typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
asum(thrust::execution_policy<DerivedPolicy>& exec,
     const Array& x)
{
    using thrust::abs;
    using std::abs;

    typedef typename Array::value_type ValueType;

    detail::absolute<ValueType> unary_op;
    thrust::plus<ValueType>     binary_op;

    ValueType init = 0;

    return abs(thrust::transform_reduce(exec, x.begin(), x.end(), unary_op, init, binary_op));
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
    size_t N = x.size();

    thrust::for_each(exec,
                     thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin())) + N,
                     detail::AXPY<ScalarType>(alpha));
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
                                 thrust::make_transform_iterator(x.begin(), detail::conjugate<OutputType>()),
                                 thrust::make_transform_iterator(x.end(),  detail::conjugate<OutputType>()),
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
    using thrust::sqrt;
    using thrust::abs;

    using std::sqrt;
    using std::abs;

    typedef typename Array::value_type ValueType;

    cusp::detail::norm_squared<ValueType> unary_op;
    thrust::plus<ValueType>   binary_op;

    ValueType init = 0;

    return sqrt( abs( thrust::transform_reduce(exec, x.begin(), x.end(), unary_op, init, binary_op) ) );
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
         typename Array2d2,
         typename Array2d3>
void trmm(thrust::execution_policy<DerivedPolicy>& exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C)
{
    throw cusp::not_implemented_exception("CUSP TRMM not implemented");
}

template<typename DerivedPolicy,
         typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void trsm(thrust::execution_policy<DerivedPolicy>& exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C)
{
    throw cusp::not_implemented_exception("CUSP TRSM not implemented");
}

template <typename DerivedPolicy,
          typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm1(thrust::execution_policy<DerivedPolicy>& exec,
     const Array& x)
{
    using thrust::abs;
    using std::abs;

    typedef typename Array::value_type ValueType;

    detail::absolute<ValueType> unary_op;
    thrust::plus<ValueType>     binary_op;

    ValueType init = 0;

    return abs(thrust::transform_reduce(exec, x.begin(), x.end(), unary_op, init, binary_op));
}

template <typename DerivedPolicy,
          typename Array>
typename Array::value_type
nrmmax(thrust::execution_policy<DerivedPolicy>& exec,
       const Array& x)
{
    typedef typename Array::value_type ValueType;

    detail::absolute<ValueType>  unary_op;
    detail::maximum<ValueType>   binary_op;

    ValueType init = 0;

    return thrust::transform_reduce(exec, x.begin(), x.end(), unary_op, init, binary_op);
}

} // end namespace thrustblas
} // end namespace blas
} // end namespace cusp


