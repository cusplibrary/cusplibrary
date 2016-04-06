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

#include <cusp/detail/config.h>
#include <cusp/detail/type_traits.h>

#include <cusp/detail/execution_policy.h>

namespace cusp
{

template <typename DerivedPolicy,
          typename SourceType,
          typename DestinationType,
          typename Format1,
          typename Format2>
void convert(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
             const SourceType& src,
             DestinationType& dst,
             Format1, Format2);

namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy,
          typename SourceType,
          typename DestinationType,
          typename Format>
void convert(thrust::execution_policy<DerivedPolicy>& exec,
             const SourceType& src,
             DestinationType& dst,
             Format&, Format&);

template <typename DerivedPolicy,
          typename SourceType,
          typename DestinationType,
          typename Format1,
          typename Format2>
typename cusp::detail::enable_if_same_system<SourceType,DestinationType>::type
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        Format1&, Format2&);

template <typename DerivedPolicy,
          typename SourceType,
          typename DestinationType,
          typename Format1,
          typename Format2>
typename cusp::detail::enable_if_different_system<SourceType,DestinationType>::type
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        Format1&, Format2&);

template <typename DerivedPolicy,
          typename SourceType,
          typename DestinationType>
void convert(thrust::execution_policy<DerivedPolicy> &exec,
             const SourceType& src,
             DestinationType& dst);

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp

#include <cusp/system/detail/generic/convert.inl>
