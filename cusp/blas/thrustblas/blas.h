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
namespace thrustblas
{

template <typename Array1,
         typename Array2,
         typename ScalarType>
void axpy(const Array1& x,
          Array2& y,
          ScalarType alpha,
          cusp::any_memory)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y);
    cusp::blas::thrustblas::detail::axpy(x.begin(), x.end(), y.begin(), alpha);
}

template <typename Array1,
         typename Array2,
         typename ScalarType>
void axpy(const Array1& x,
          const Array2& y,
          ScalarType alpha,
          cusp::any_memory)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y);
    cusp::blas::thrustblas::detail::axpy(x.begin(), x.end(), y.begin(), alpha);
}

template <typename Array1,
         typename Array2,
         typename Array3,
         typename ScalarType1,
         typename ScalarType2>
void axpby(const Array1& x,
           const Array2& y,
           Array3& z,
           ScalarType1 alpha,
           ScalarType2 beta,
           cusp::any_memory)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y, z);
    cusp::blas::thrustblas::detail::axpby(x.begin(), x.end(), y.begin(), z.begin(), alpha, beta);
}

template <typename Array1,
         typename Array2,
         typename Array3,
         typename ScalarType1,
         typename ScalarType2>
void axpby(const Array1& x,
           const Array2& y,
           const Array3& z,
           ScalarType1 alpha,
           ScalarType2 beta,
           cusp::any_memory)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y, z);
    cusp::blas::thrustblas::detail::axpby(x.begin(), x.end(), y.begin(), z.begin(), alpha, beta);
}

template <typename Array1,
         typename Array2,
         typename Array3,
         typename Array4,
         typename ScalarType1,
         typename ScalarType2,
         typename ScalarType3>
void axpbypcz(const Array1& x,
              const Array2& y,
              const Array3& z,
              Array4& output,
              ScalarType1 alpha,
              ScalarType2 beta,
              ScalarType3 gamma,
              cusp::any_memory)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y, z, output);
    cusp::blas::thrustblas::detail::axpbypcz(x.begin(), x.end(), y.begin(), z.begin(), output.begin(), alpha, beta, gamma);
}

template <typename Array1,
         typename Array2,
         typename Array3,
         typename Array4,
         typename ScalarType1,
         typename ScalarType2,
         typename ScalarType3>
void axpbypcz(const Array1& x,
              const Array2& y,
              const Array3& z,
              const Array4& output,
              ScalarType1 alpha,
              ScalarType2 beta,
              ScalarType3 gamma,
              cusp::any_memory)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y, z, output);
    cusp::blas::thrustblas::detail::axpbypcz(x.begin(), x.end(), y.begin(), z.begin(), output.begin(), alpha, beta, gamma);
}

template <typename Array1,
         typename Array2,
         typename Array3>
void xmy(const Array1& x,
         const Array2& y,
         Array3& output,
         cusp::any_memory)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y, output);
    cusp::blas::thrustblas::detail::xmy(x.begin(), x.end(), y.begin(), output.begin());
}

template <typename Array1,
         typename Array2,
         typename Array3>
void xmy(const Array1& x,
         const Array2& y,
         const Array3& output,
         cusp::any_memory)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y, output);
    cusp::blas::thrustblas::detail::xmy(x.begin(), x.end(), y.begin(), output.begin());
}

template <typename Array1,
         typename Array2>
void copy(const Array1& x,
          Array2& y,
          cusp::any_memory)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y);
    cusp::blas::thrustblas::detail::copy(x.begin(), x.end(), y.begin());
}

template <typename Array1,
         typename Array2>
void copy(const Array1& x,
          const Array2& y,
          cusp::any_memory)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y);
    cusp::blas::thrustblas::detail::copy(x.begin(), x.end(), y.begin());
}

// TODO properly harmonize heterogenous types
template <typename Array1,
         typename Array2>
typename Array1::value_type
dot(const Array1& x,
    const Array2& y,
    cusp::any_memory)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y);
    return cusp::blas::thrustblas::detail::dot(x.begin(), x.end(), y.begin());
}

// TODO properly harmonize heterogenous types
template <typename Array1,
         typename Array2>
typename Array1::value_type
dotc(const Array1& x,
     const Array2& y,
     cusp::any_memory)
{
    CUSP_PROFILE_SCOPED();
    detail::assert_same_dimensions(x, y);
    return cusp::blas::thrustblas::detail::dotc(x.begin(), x.end(), y.begin());
}

template <typename Array,
         typename ScalarType>
void fill(Array& x,
          ScalarType alpha,
          cusp::any_memory)
{
    CUSP_PROFILE_SCOPED();
    cusp::blas::thrustblas::detail::fill(x.begin(), x.end(), alpha);
}

template <typename Array,
         typename ScalarType>
void fill(const Array& x,
          ScalarType alpha,
          cusp::any_memory)
{
    CUSP_PROFILE_SCOPED();
    cusp::blas::thrustblas::detail::fill(x.begin(), x.end(), alpha);
}

template <typename Array>
typename norm_type<typename Array::value_type>::type
nrm1(const Array& x,
     cusp::any_memory)
{
    CUSP_PROFILE_SCOPED();
    return cusp::blas::thrustblas::detail::nrm1(x.begin(), x.end());
}

template <typename Array>
typename norm_type<typename Array::value_type>::type
nrm2(const Array& x,
     cusp::any_memory)
{
    CUSP_PROFILE_SCOPED();
    return cusp::blas::thrustblas::detail::nrm2(x.begin(), x.end());
}

template <typename Array>
typename Array::value_type
nrmmax(const Array& x,
       cusp::any_memory)
{
    CUSP_PROFILE_SCOPED();
    return cusp::blas::thrustblas::detail::nrmmax(x.begin(), x.end());
}

template <typename Array,
         typename ScalarType>
void scal(Array& x,
          ScalarType alpha,
          cusp::any_memory)
{
    CUSP_PROFILE_SCOPED();
    cusp::blas::thrustblas::detail::scal(x.begin(), x.end(), alpha);
}

template <typename Array,
         typename ScalarType>
void scal(const Array& x,
          ScalarType alpha,
          cusp::any_memory)
{
    CUSP_PROFILE_SCOPED();
    cusp::blas::thrustblas::detail::scal(x.begin(), x.end(), alpha);
}
} // end namespace thrustblas
} // end namespace blas
} // end namespace cusp

