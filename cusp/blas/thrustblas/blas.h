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
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm1(const thrustblas::detail::blas_policy<typename Array::memory_space>& policy, Array& x)
{
    using thrust::abs;
    using std::abs;

    typedef typename Array::value_type ValueType;

    detail::absolute<ValueType> unary_op;
    thrust::plus<ValueType>     binary_op;

    ValueType init = 0;

    return abs(thrust::transform_reduce(x.begin(), x.end(), unary_op, init, binary_op));
}

template <typename Array>
typename cusp::detail::norm_type<typename Array::value_type>::type
nrm2(const thrustblas::detail::blas_policy<typename Array::memory_space>& policy, Array& x)
{
    using thrust::sqrt;
    using thrust::abs;

    using std::sqrt;
    using std::abs;

    typedef typename Array::value_type ValueType;

    cusp::detail::norm_squared<ValueType> unary_op;
    thrust::plus<ValueType>   binary_op;

    ValueType init = 0;

    return sqrt( abs(thrust::transform_reduce(x.begin(), x.end(), unary_op, init, binary_op)) );
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
  throw cusp::not_implemented_exception("CUSP GEMV not implemented");
}

template<typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void gemm(const thrustblas::detail::blas_policy<typename Array2d3::memory_space>& policy,
          const Array2d1& A,
          const Array2d2& B,
          Array2d3& C)
{
  throw cusp::not_implemented_exception("CUSP GEMM not implemented");
}

} // end namespace blas
} // end namespace cusp

