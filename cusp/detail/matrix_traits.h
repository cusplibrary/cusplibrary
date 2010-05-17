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

#include <cusp/detail/config.h>

#include <thrust/detail/type_traits.h>

namespace cusp
{
namespace detail
{

using thrust::detail::true_type;
using thrust::detail::false_type;

template <typename T>
class is_coo_matrix;

template <typename T>
class is_csr_matrix;

template <typename T>
class is_dia_matrix;

template <typename T>
class is_ell_matrix;

template <typename T>
class is_hyb_matrix;

} // end namespace detail
} // end namespace cusp

#include <cusp/detail/matrix_traits.inl>

