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

#include <cusp/format.h>
// TODO replace with detail/array2d_utils.h or something
#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>

#include <cusp/detail/utils.h>
#include <cusp/detail/format_utils.h>

#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

namespace cusp
{
namespace detail
{

// COO format
template <typename MatrixType1,   typename MatrixType2>
void transpose(const MatrixType1& A, MatrixType2& At,
               cusp::coo_format,
               cusp::coo_format)
{
    At.resize(A.num_cols, A.num_rows, A.num_entries);

    cusp::copy(A.row_indices,    At.column_indices);
    cusp::copy(A.column_indices, At.row_indices);
    cusp::copy(A.values,         At.values);

    At.sort_by_row();
}


// CSR format
template <typename MatrixType1,   typename MatrixType2>
void transpose(const MatrixType1& A, MatrixType2& At,
               cusp::csr_format,
               cusp::csr_format)
{
    typedef typename MatrixType2::index_type   IndexType2;
    typedef typename MatrixType2::memory_space MemorySpace2;

    At.resize(A.num_cols, A.num_rows, A.num_entries);

    cusp::detail::offsets_to_indices(A.row_offsets, At.column_indices);
    cusp::copy(A.values, At.values);

    cusp::array1d<IndexType2,MemorySpace2> At_row_indices(A.column_indices);

    cusp::detail::sort_by_row(At_row_indices, At.column_indices, At.values);
    
    cusp::detail::indices_to_offsets(At_row_indices, At.row_offsets);
}


// convert logical linear index in the (tranposed) destination into a physical index in the source
template <typename IndexType, typename Orientation1, typename Orientation2>
struct transpose_index_functor : public thrust::unary_function<IndexType,IndexType>
{
  IndexType num_rows, num_cols, pitch; // source dimensions

  transpose_index_functor(IndexType num_rows, IndexType num_cols, IndexType pitch)
    : num_rows(num_rows), num_cols(num_cols), pitch(pitch) {}

  __host__ __device__
  IndexType operator()(IndexType linear_index)
  {
      IndexType i = cusp::detail::linear_index_to_row_index(linear_index, num_cols, num_rows, Orientation2());
      IndexType j = cusp::detail::linear_index_to_col_index(linear_index, num_cols, num_rows, Orientation2());
      
      return cusp::detail::index_of(j, i, pitch, Orientation1());
  }
};

// Array2d format 
template <typename MatrixType1,   typename MatrixType2>
void transpose(const MatrixType1& A, MatrixType2& At,
               cusp::array2d_format,
               cusp::array2d_format)
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


// Default case uses CSR transpose
template <typename MatrixType1,   typename MatrixType2,
          typename MatrixFormat1, typename MatrixFormat2>
void transpose(const MatrixType1& A, MatrixType2& At,
                     MatrixFormat1,  MatrixFormat2)
{
    typedef typename MatrixType1::index_type   IndexType;
    typedef typename MatrixType1::value_type   ValueType;
    typedef typename MatrixType1::memory_space MemorySpace;

    cusp::csr_matrix<IndexType, ValueType, MemorySpace> A_csr(A);
    cusp::csr_matrix<IndexType, ValueType, MemorySpace> At_csr;
    cusp::transpose(A_csr, At_csr);

    cusp::convert(At_csr, At);
}

} // end namespace detail

template <typename MatrixType1, typename MatrixType2>
void transpose(const MatrixType1& A, MatrixType2& At)
{
    CUSP_PROFILE_SCOPED();

    cusp::detail::transpose(A, At,
                            typename MatrixType1::format(),
                            typename MatrixType2::format());
}

} // end namespace cusp

