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
#include <cusp/copy.h>
#include <cusp/convert.h>
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

// Array2d format
template <typename DerivedPolicy, typename MatrixType1, typename MatrixType2>
void transpose(cusp::execution_policy<DerivedPolicy>& exec,
               const MatrixType1& A, MatrixType2& At,
               cusp::array2d_format, cusp::array2d_format)
{
    typedef typename MatrixType1::orientation Orientation1;
    typedef typename MatrixType2::orientation Orientation2;

    At.resize(A.num_cols, A.num_rows);

    thrust::counting_iterator<size_t> begin(0);
    thrust::counting_iterator<size_t> end(A.num_entries);

    // prefer coalesced writes to coalesced reads
    cusp::detail::transpose_index_functor    <size_t, Orientation1, Orientation2> func1(A.num_rows,  A.num_cols,  A.pitch);
    cusp::detail::logical_to_physical_functor<size_t, Orientation2>               func2(At.num_rows, At.num_cols, At.pitch);

    thrust::copy(thrust::make_permutation_iterator(A.values.begin(),  thrust::make_transform_iterator(begin, func1)),
                 thrust::make_permutation_iterator(A.values.begin(),  thrust::make_transform_iterator(end,   func1)),
                 thrust::make_permutation_iterator(At.values.begin(), thrust::make_transform_iterator(begin, func2)));
}

// COO format
template <typename DerivedPolicy, typename MatrixType1, typename MatrixType2>
void transpose(cusp::execution_policy<DerivedPolicy>& exec,
               const MatrixType1& A, MatrixType2& At,
               cusp::coo_format, cusp::coo_format)
{
    At.resize(A.num_cols, A.num_rows, A.num_entries);

    cusp::copy(A.row_indices,    At.column_indices);
    cusp::copy(A.column_indices, At.row_indices);
    cusp::copy(A.values,         At.values);

    At.sort_by_row();
}

// CSR format
template <typename DerivedPolicy, typename MatrixType1, typename MatrixType2>
void transpose(cusp::execution_policy<DerivedPolicy>& exec,
               const MatrixType1& A, MatrixType2& At,
               cusp::csr_format, cusp::csr_format)
{
    typedef typename MatrixType2::index_type   IndexType2;
    typedef typename MatrixType2::memory_space MemorySpace2;

    std::cout << "Executing generic transpose" << std::endl;

    At.resize(A.num_cols, A.num_rows, A.num_entries);

    cusp::detail::offsets_to_indices(A.row_offsets, At.column_indices);
    cusp::copy(A.values, At.values);

    cusp::array1d<IndexType2,MemorySpace2> At_row_indices(A.column_indices);

    cusp::detail::sort_by_row(At_row_indices, At.column_indices, At.values);

    cusp::detail::indices_to_offsets(At_row_indices, At.row_offsets);
}

// Default case uses CSR transpose
template <typename DerivedPolicy,
          typename MatrixType1, typename MatrixType2,
          typename MatrixFormat1, typename MatrixFormat2>
void transpose(cusp::execution_policy<DerivedPolicy>& exec,
               const MatrixType1& A, MatrixType2& At,
               MatrixFormat1,  MatrixFormat2)
{
    typedef typename MatrixType1::index_type   IndexType;
    typedef typename MatrixType1::value_type   ValueType;
    typedef typename MatrixType1::memory_space MemorySpace;

    std::cout << "Converting..." << std::endl;

    cusp::csr_matrix<IndexType, ValueType, MemorySpace> A_csr(A);
    cusp::csr_matrix<IndexType, ValueType, MemorySpace> At_csr;
    cusp::transpose(exec, A_csr, At_csr);

    cusp::convert(At_csr, At);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp
