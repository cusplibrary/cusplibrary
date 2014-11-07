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

#include <thrust/detail/config.h>
#include <thrust/system/detail/generic/select_system.h>

#include <cusp/multiply.h>

#include <cusp/detail/type_traits.h>
#include <cusp/system/detail/adl/multiply.h>
#include <cusp/system/detail/generic/multiply.h>

namespace cusp
{
namespace detail
{

template <typename DerivedPolicy,
         typename LinearOperator,
         typename MatrixOrVector1,
         typename MatrixOrVector2>
void multiply(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
              LinearOperator&  A,
              MatrixOrVector1& B,
              MatrixOrVector2& C,
              thrust::detail::false_type)
{
    // user-defined LinearOperator
    A(B,C);
}

template <typename DerivedPolicy,
         typename LinearOperator,
         typename MatrixOrVector1,
         typename MatrixOrVector2>
void multiply(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
              LinearOperator&  A,
              MatrixOrVector1& B,
              MatrixOrVector2& C,
              thrust::detail::true_type)
{
    using cusp::system::detail::generic::multiply;

    typedef typename LinearOperator::value_type ValueType;

    typename LinearOperator::format  format1;
    typename MatrixOrVector1::format format2;
    typename MatrixOrVector2::format format3;

    cusp::detail::zero_function<ValueType> initialize;
    thrust::multiplies<ValueType> combine;
    thrust::plus<ValueType> reduce;

    cusp::detail::multiply(thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
                           A, B, C, initialize, combine, reduce, format1, format2, format3);
}

} // end namespace detail

template <typename DerivedPolicy,
         typename LinearOperator,
         typename MatrixOrVector1,
         typename MatrixOrVector2>
void multiply(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
              LinearOperator&  A,
              MatrixOrVector1& B,
              MatrixOrVector2& C)
{
    typename thrust::detail::is_convertible<typename LinearOperator::format,known_format>::type is_known_operator;

    cusp::detail::multiply(exec, A, B, C, is_known_operator);
}

template <typename LinearOperator,
         typename MatrixOrVector1,
         typename MatrixOrVector2>
void multiply(LinearOperator&  A,
              MatrixOrVector1& B,
              MatrixOrVector2& C)
{
    using thrust::system::detail::generic::select_system;

    typedef typename LinearOperator::memory_space  System1;
    typedef typename MatrixOrVector1::memory_space System2;
    typedef typename MatrixOrVector2::memory_space System3;

    System1 system1;
    System2 system2;
    System3 system3;

    cusp::multiply(select_system(system1,system2,system3), A, B, C);
}

} // end namespace cusp

