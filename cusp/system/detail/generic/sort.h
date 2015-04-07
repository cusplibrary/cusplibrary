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

/*! \file transpose.h
 *  \brief Sequential implementations of transpose algorithms.
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/array1d.h>

#include <thrust/sort.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy, typename ArrayType>
void counting_sort(thrust::execution_policy<DerivedPolicy>& exec,
                   ArrayType& keys, typename ArrayType::value_type min, typename ArrayType::value_type max)
{
  thrust::stable_sort(exec, keys.begin(), keys.end());
}

template <typename DerivedPolicy, typename ArrayType1, typename ArrayType2>
void counting_sort_by_key(thrust::execution_policy<DerivedPolicy>& exec,
                          ArrayType1& keys, ArrayType2& vals,
                          typename ArrayType1::value_type min, typename ArrayType1::value_type max)
{
  thrust::stable_sort_by_key(exec, keys.begin(), keys.end(), vals.begin());
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp
