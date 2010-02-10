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


#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>

#include <cusp/detail/utils.h>
#include <cusp/detail/format_utils.h>

#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/iterator/transform_iterator.h>

namespace cusp
{
namespace detail
{

template <typename IndexType1, typename ValueType1, typename MemorySpace1,
          typename IndexType2, typename ValueType2, typename MemorySpace2>
void transpose(const cusp::coo_matrix<IndexType1,ValueType1,MemorySpace1>& A,
                     cusp::coo_matrix<IndexType2,ValueType2,MemorySpace2>& At)
{
    cusp::coo_matrix<IndexType2,ValueType2,MemorySpace2> temp(A.num_cols, A.num_rows, A.num_entries);

    cusp::array1d<IndexType2,MemorySpace2> permutation(A.num_entries);
    thrust::sequence(permutation.begin(), permutation.end());

    temp.row_indices = A.column_indices;

    thrust::stable_sort_by_key(temp.row_indices.begin(), temp.row_indices.end(), permutation.begin());

    // XXX could be fused with a zip_iterator
    thrust::next::gather(permutation.begin(), permutation.end(),
                         A.row_indices.begin(),
                         temp.column_indices.begin());
    
    thrust::next::gather(permutation.begin(), permutation.end(),
                         A.values.begin(),
                         temp.values.begin());

    At.swap(temp);
}

    
template <typename IndexType1, typename ValueType1, typename MemorySpace1,
          typename IndexType2, typename ValueType2, typename MemorySpace2>
void transpose(const cusp::csr_matrix<IndexType1,ValueType1,MemorySpace1>& A,
                     cusp::csr_matrix<IndexType2,ValueType2,MemorySpace2>& At)
{
    cusp::csr_matrix<IndexType2,ValueType2,MemorySpace2> temp(A.num_cols, A.num_rows, A.num_entries);
    
    cusp::array1d<IndexType2,MemorySpace2> permutation(A.num_entries);
    thrust::sequence(permutation.begin(), permutation.end());
    
    // sort column indices of A
    cusp::array1d<IndexType2,MemorySpace2> indices(A.column_indices);
    thrust::stable_sort_by_key(indices.begin(), indices.end(), permutation.begin());

    // compute row offsets of At
    cusp::detail::indices_to_offsets(indices, temp.row_offsets);

    // compute row indices of A
    cusp::detail::offsets_to_indices(A.row_offsets, indices);

    // XXX could be fused with a zip_iterator
    thrust::next::gather(permutation.begin(), permutation.end(),
                         indices.begin(),
                         temp.column_indices.begin());
    
    thrust::next::gather(permutation.begin(), permutation.end(),
                         A.values.begin(),
                         temp.values.begin());

    At.swap(temp);
}


// convert a linear index to a linear index in the transpose
template <typename T, typename SourceOrientation, typename DestinationOrientation>
struct transpose_index : public thrust::unary_function<T, T>
{
    T m, n; // destination dimensions

    __host__ __device__
    transpose_index(T _m, T _n) : m(_m), n(_n) {}

    __host__ __device__
    T operator()(T linear_index)
    {
        T i = cusp::detail::linear_index_to_row_index(linear_index, m, n, DestinationOrientation());
        T j = cusp::detail::linear_index_to_col_index(linear_index, m, n, DestinationOrientation());

        return cusp::detail::index_of(j, i, n, m, SourceOrientation());
    }
};

    
template <typename ValueType1, typename MemorySpace1, typename SourceOrientation,
          typename ValueType2, typename MemorySpace2, typename DestinationOrientation>
void transpose(const cusp::array2d<ValueType1,MemorySpace1,SourceOrientation>& A,
                     cusp::array2d<ValueType2,MemorySpace2,DestinationOrientation>& At)
{
    At.resize(A.num_cols, A.num_rows);

    thrust::counting_iterator<size_t> begin(0);
    thrust::counting_iterator<size_t> end(A.values.size());

    thrust::next::gather(thrust::make_transform_iterator(begin, transpose_index<size_t, SourceOrientation, DestinationOrientation>(At.num_rows, At.num_cols)),
                         thrust::make_transform_iterator(end,   transpose_index<size_t, SourceOrientation, DestinationOrientation>(At.num_rows, At.num_cols)),
                         A.values.begin(),
                         At.values.begin());
}


// all other formats go through CSR
template <typename MatrixType1, typename MatrixType2>
void transpose(const MatrixType1& A, MatrixType2& At)
{
    typedef typename MatrixType1::index_type   IndexType;
    typedef typename MatrixType1::value_type   ValueType;
    typedef typename MatrixType1::memory_space MemorySpace;

    cusp::csr_matrix<IndexType, ValueType, MemorySpace> A_csr(A);
    cusp::csr_matrix<IndexType, ValueType, MemorySpace> At_csr;
    cusp::detail::transpose(A_csr, At_csr);

    At = At_csr;
}

} // end namespace detail

template <typename MatrixType1, typename MatrixType2>
void transpose(const MatrixType1& A, MatrixType2& At)
{
    cusp::detail::transpose(A, At);
}

} // end namespace cusp

