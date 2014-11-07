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

#include <cusp/detail/config.h>

#include <thrust/execution_policy.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy,
          typename LinearOperator, typename MatrixOrVector1, typename MatrixOrVector2,
          typename UnaryFunction,  typename BinaryFunction1, typename BinaryFunction2,
          typename Format1, typename Format2, typename Format3>
void multiply(thrust::detail::execution_policy_base<DerivedPolicy> &exec,
              LinearOperator&  A,
              MatrixOrVector1& B,
              MatrixOrVector2& C,
              UnaryFunction  initialize, BinaryFunction1 combine, BinaryFunction2 reduce,
              Format1&, Format2&, Format3&);

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp

#include <cusp/system/detail/generic/multiply.inl>
