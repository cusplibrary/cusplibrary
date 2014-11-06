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

#include <cusp/format.h>
#include <cusp/array1d.h>

#include <cusp/system/detail/sequential/execution_policy.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace sequential
{

using namespace thrust::system::detail::sequential;

//////////////
// COO SpMV //
//////////////
template <typename Matrix,
         typename Vector1,
         typename Vector2,
         typename UnaryFunction,
         typename BinaryFunction1,
         typename BinaryFunction2>
void multiply(sequential::execution_policy<DerivedPolicy>& exec,
              const Matrix&  A,
              const Vector1& x,
              Vector2& y,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Vector2::value_type ValueType;

    for(size_t i = 0; i < A.num_rows; i++)
        y[i] = initialize(y[i]);

    for(size_t n = 0; n < A.num_entries; n++)
    {
        const IndexType& i   = A.row_indices[n];
        const IndexType& j   = A.column_indices[n];
        const ValueType& Aij = A.values[n];
        const ValueType& xj  = x[j];

        y[i] = reduce(y[i], combine(Aij, xj));
    }
}

template <typename Matrix,
         typename Vector1,
         typename Vector2>
void multiply(const Matrix&  A,
              const Vector1& x,
              Vector2& y)
{
    typedef typename Vector2::value_type ValueType;

    spmv_coo(A, x, y,
             cusp::detail::zero_function<ValueType>(),
             thrust::multiplies<ValueType>(),
             thrust::plus<ValueType>());
}

} // end namespace sequential
} // end namespace detail
} // end namespace system
} // end namespace cusp

