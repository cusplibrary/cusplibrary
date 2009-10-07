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

#include <cusp/csr_matrix.h>


namespace cusp
{

namespace detail
{

namespace host
{

template <typename IndexType, typename ValueType, class BinaryOperator>
void elementwise_operation(      cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& C,
                           const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& A,
                           const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& B,
                           BinaryOperator op);

} // end namespace host

} // end namespace detail

} // end namespace cusp

#include <cusp/detail/host/detail/elementwise_operation.inl>

