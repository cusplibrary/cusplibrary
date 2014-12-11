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

#include <thrust/execution_policy.h>

namespace cusp
{

/* \cond */
template <typename DerivedPolicy, typename OffsetArray, typename IndexArray>
void offsets_to_indices(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                        const OffsetArray& offsets,
                        IndexArray& indices);
/* \endcond */

template <typename OffsetArray, typename IndexArray>
void offsets_to_indices(const OffsetArray& offsets,
                        IndexArray& indices);

/* \cond */
template <typename DerivedPolicy, typename IndexArray, typename OffsetArray>
void indices_to_offsets(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                        const IndexArray& indices,
                        OffsetArray& offsets);
/* \endcond */

template <typename IndexArray, typename OffsetArray>
void indices_to_offsets(const IndexArray& indices,
                        OffsetArray& offsets);

/* \cond */
template <typename DerivedPolicy, typename MatrixType, typename ArrayType>
void extract_diagonal(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                      const MatrixType& A,
                      ArrayType& output);
/* \endcond */

template <typename MatrixType, typename ArrayType>
void extract_diagonal(const MatrixType& A, ArrayType& output);

/* \cond */
template <typename DerivedPolicy, typename ArrayType1, typename ArrayType2>
size_t count_diagonals(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                       const size_t num_rows,
                       const size_t num_cols,
                       const ArrayType1& row_indices,
                       const ArrayType2& column_indices );
/* \endcond */

template <typename ArrayType1, typename ArrayType2>
size_t count_diagonals(const size_t num_rows,
                       const size_t num_cols,
                       const ArrayType1& row_indices,
                       const ArrayType2& column_indices);

/* \cond */
template <typename DerivedPolicy, typename ArrayType>
size_t compute_max_entries_per_row(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                   const ArrayType& row_offsets);
/* \endcond */

template <typename ArrayType>
size_t compute_max_entries_per_row(const ArrayType& row_offsets);

/* \cond */
template <typename DerivedPolicy, typename ArrayType>
size_t compute_optimal_entries_per_row(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                       const ArrayType& row_offsets,
                                       float relative_speed = 3.0f,
                                       size_t breakeven_threshold = 4096);
/* \endcond */

template <typename ArrayType>
size_t compute_optimal_entries_per_row(const ArrayType& row_offsets,
                                       float relative_speed = 3.0f,
                                       size_t breakeven_threshold = 4096);
} // end namespace cusp

#include <cusp/detail/format_utils.inl>

