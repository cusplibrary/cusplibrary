/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

#include <thrust/detail/config.h>

#include <cusp/detail/format.h>
#include <cusp/detail/functional.h>

#include <cusp/system/detail/generic/multiply/generalized_spmv.h>
#include <cusp/system/detail/generic/multiply/permute.h>
#include <cusp/system/detail/generic/multiply/spgemm.h>
#include <cusp/system/detail/generic/multiply/spmv.h>

#include <thrust/functional.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy,
          typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(thrust::execution_policy<DerivedPolicy> &exec,
              const LinearOperator&  A,
              const MatrixOrVector1& B,
              MatrixOrVector2& C,
              unknown_format,
              array1d_format,
              array1d_format)
{
    // user-defined LinearOperator
    const_cast<LinearOperator&>(A)(B,C);
}

template <typename DerivedPolicy,
          typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(thrust::execution_policy<DerivedPolicy> &exec,
              const  LinearOperator&  A,
              const  MatrixOrVector1& B,
              MatrixOrVector2& C,
              known_format,
              array1d_format,
              array1d_format)
{
    typedef typename LinearOperator::value_type ValueType;

    cusp::detail::zero_function<ValueType> initialize;
    thrust::multiplies<ValueType> combine;
    thrust::plus<ValueType> reduce;

    cusp::multiply(exec, A, B, C, initialize, combine, reduce);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp

