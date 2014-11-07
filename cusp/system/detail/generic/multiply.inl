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
#include <thrust/system/detail/generic/tag.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cusp/format.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>

#include <cusp/detail/utils.h>
#include <cusp/detail/array2d_format_utils.h>

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
          typename UnaryFunction,  typename BinaryFunction1, typename BinaryFunction2>
void multiply(thrust::execution_policy<DerivedPolicy> &exec,
              LinearOperator&  A,
              MatrixOrVector1& B,
              MatrixOrVector2& C,
              UnaryFunction  initialize, BinaryFunction1 combine, BinaryFunction2 reduce,
              cusp::permutation_format, cusp::array1d_format, cusp::array1d_format)
{
    thrust::gather(A.permutation.begin(), A.permutation.end(), B.begin(), C.begin());
}

template <typename DerivedPolicy,
          typename LinearOperator, typename MatrixOrVector1, typename MatrixOrVector2,
          typename UnaryFunction,  typename BinaryFunction1, typename BinaryFunction2>
void multiply(thrust::execution_policy<DerivedPolicy> &exec,
              LinearOperator&  A,
              MatrixOrVector1& B,
              MatrixOrVector2& C,
              UnaryFunction  initialize, BinaryFunction1 combine, BinaryFunction2 reduce,
              hyb_format&, array1d_format&, array1d_format&)
{
    typedef typename MatrixOrVector2::value_type ValueType;

    cusp::multiply(exec, A.ell, B, C, initialize, combine, reduce);
    cusp::multiply(exec, A.coo, B, C, thrust::identity<ValueType>(), combine, reduce);
}

template <typename DerivedPolicy,
          typename LinearOperator, typename MatrixOrVector1, typename MatrixOrVector2,
          typename UnaryFunction,  typename BinaryFunction1, typename BinaryFunction2>
void multiply(thrust::execution_policy<DerivedPolicy> &exec,
              LinearOperator&  A,
              MatrixOrVector1& B,
              MatrixOrVector2& C,
              UnaryFunction  initialize, BinaryFunction1 combine, BinaryFunction2 reduce,
              csr_format&, array1d_format&, array1d_format&)
{
  std::cout << " Calling generic spmv " << std::endl;
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp
