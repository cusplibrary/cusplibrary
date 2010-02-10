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

#include <cusp/coo_matrix.h>
#include <cusp/array2d.h>

namespace cusp
{
namespace detail
{
namespace generic
{

template <typename IndexType,
          typename ValueType,
          typename MemorySpace>
void multiply(const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& A,
              const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& B,
                    cusp::coo_matrix<IndexType,ValueType,MemorySpace>& C);

template <typename ValueType,
          typename MemorySpace>
void multiply(const cusp::array2d<ValueType,MemorySpace>& A,
              const cusp::array2d<ValueType,MemorySpace>& B,
                    cusp::array2d<ValueType,MemorySpace>& C);

} // end namespace generic
} // end namespace detail
} // end namespace cusp

#include <cusp/detail/generic/detail/multiply.inl>

