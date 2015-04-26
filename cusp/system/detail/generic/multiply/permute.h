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

#include <thrust/detail/config.h>

#include <cusp/detail/format.h>
#include <cusp/copy.h>

#include <thrust/gather.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy,
          typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(thrust::execution_policy<DerivedPolicy> &exec,
              const LinearOperator&  A,
              const MatrixOrVector1& B,
              MatrixOrVector2& C,
              permutation_format,
              array1d_format,
              array1d_format)
{
    C.resize(B.size());

    thrust::scatter(exec, B.begin(), B.end(), A.permutation.begin(), C.begin());
}

template <typename DerivedPolicy,
          typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(thrust::execution_policy<DerivedPolicy> &exec,
              const LinearOperator&  A,
              const MatrixOrVector1& B,
              MatrixOrVector2& C,
              permutation_format,
              coo_format,
              coo_format)
{
    C.resize(B.num_rows, B.num_cols, B.num_entries);

    thrust::gather(exec, B.row_indices.begin(), B.row_indices.end(), A.permutation.begin(), C.row_indices.begin());

    cusp::copy(exec, B.column_indices, C.column_indices);
    cusp::copy(exec, B.values, C.values);

    C.sort_by_row();
}

template <typename DerivedPolicy,
          typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(thrust::execution_policy<DerivedPolicy> &exec,
              const LinearOperator&  A,
              const MatrixOrVector1& B,
              MatrixOrVector2& C,
              coo_format,
              permutation_format,
              coo_format)
{
    C.resize(A.num_rows, A.num_cols, A.num_entries);

    thrust::gather(exec, A.column_indices.begin(), A.column_indices.end(), B.permutation.begin(), C.column_indices.begin());

    cusp::copy(exec, A.row_indices, C.row_indices);
    cusp::copy(exec, A.values, C.values);

    C.sort_by_row_and_column();
}

template <typename DerivedPolicy,
          typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(thrust::execution_policy<DerivedPolicy> &exec,
              const LinearOperator&  A,
              const MatrixOrVector1& B,
              MatrixOrVector2& C,
              permutation_format,
              sparse_format,
              sparse_format)
{
   typename cusp::detail::as_coo_type<MatrixOrVector1>::type B_(B);
   typename cusp::detail::as_coo_type<MatrixOrVector2>::type C_(C);

   cusp::multiply(exec, A, B_, C_);
   cusp::convert(exec, C_, C);
}

template <typename DerivedPolicy,
          typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(thrust::execution_policy<DerivedPolicy> &exec,
              const LinearOperator&  A,
              const MatrixOrVector1& B,
              MatrixOrVector2& C,
              sparse_format,
              permutation_format,
              sparse_format)
{
   typename cusp::detail::as_coo_type<MatrixOrVector1>::type A_(A);
   typename cusp::detail::as_coo_type<MatrixOrVector2>::type C_(C);

   cusp::multiply(exec, A_, B, C_);
   cusp::convert(exec, C_, C);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp
