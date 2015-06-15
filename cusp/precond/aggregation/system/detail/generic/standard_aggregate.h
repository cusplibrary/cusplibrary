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

#include <cusp/detail/config.h>

#include <cusp/execution_policy.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{

template <typename MatrixType, typename ArrayType>
void standard_aggregate(const MatrixType& A, ArrayType& aggregates, ArrayType& roots);

namespace detail
{

template <typename DerivedPolicy, typename MatrixType, typename ArrayType>
void standard_aggregate(thrust::system::detail::sequential::execution_policy<DerivedPolicy> &exec,
                        const MatrixType& A, ArrayType& S, ArrayType& roots);

template <typename DerivedPolicy, typename MatrixType, typename ArrayType>
void standard_aggregate(thrust::execution_policy<DerivedPolicy> &exec,
                        const MatrixType& A, ArrayType& S, ArrayType& roots);

} // end namespace detail
} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

#include <cusp/precond/aggregation/system/detail/sequential/standard_aggregate.inl>
#include <cusp/precond/aggregation/system/detail/generic/standard_aggregate.inl>
