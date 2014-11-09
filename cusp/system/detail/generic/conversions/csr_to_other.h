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

#include <cusp/copy.h>
#include <cusp/format.h>
#include <cusp/sort.h>

#include <cusp/blas/blas.h>
#include <cusp/detail/format_utils.h>

#include <thrust/count.h>
#include <thrust/gather.h>
#include <thrust/inner_product.h>
#include <thrust/replace.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <cassert>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy,
          typename SourceType,
          typename DestinationType>
void csr_to_coo(thrust::execution_policy<DerivedPolicy>& exec,
                const SourceType& src, DestinationType& dst)
{
    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    cusp::detail::offsets_to_indices(src.row_offsets, dst.row_indices);
    cusp::copy(src.column_indices, dst.column_indices);
    cusp::copy(src.values,         dst.values);
}


template <typename DerivedPolicy,
          typename SourceType,
          typename DestinationType>
void csr_to_dia(thrust::execution_policy<DerivedPolicy>& exec,
                const SourceType& src, DestinationType& dst,
                const size_t alignment)
{
    typedef typename DestinationType::index_type IndexType;
    typedef typename DestinationType::value_type ValueType;
    typedef typename DestinationType::memory_space MemorySpace;

    // compute number of occupied diagonals and enumerate them
    cusp::array1d<IndexType,MemorySpace> row_indices(src.num_entries);
    cusp::detail::offsets_to_indices(src.row_offsets, row_indices);

    cusp::array1d<IndexType,MemorySpace> diag_map(src.num_entries);
    thrust::transform(thrust::make_zip_iterator( thrust::make_tuple( row_indices.begin(), src.column_indices.begin() ) ),
                      thrust::make_zip_iterator( thrust::make_tuple( row_indices.end()  , src.column_indices.end() ) )  ,
                      diag_map.begin(),
                      occupied_diagonal_functor<IndexType>(src.num_rows));

    // place ones in diagonals array locations with occupied diagonals
    cusp::array1d<IndexType,MemorySpace> diagonals(src.num_rows+src.num_cols,IndexType(0));
    thrust::scatter(thrust::constant_iterator<IndexType>(1),
                    thrust::constant_iterator<IndexType>(1)+src.num_entries,
                    diag_map.begin(),
                    diagonals.begin());

    const IndexType num_diagonals = thrust::reduce(diagonals.begin(), diagonals.end());

    // allocate DIA structure
    dst.resize(src.num_rows, src.num_cols, src.num_entries, num_diagonals, alignment);

    // fill in values array
    thrust::fill(dst.values.values.begin(), dst.values.values.end(), ValueType(0));

    // fill in diagonal_offsets array
    thrust::copy_if(thrust::counting_iterator<IndexType>(0),
                    thrust::counting_iterator<IndexType>(src.num_rows+src.num_cols),
                    diagonals.begin(),
                    dst.diagonal_offsets.begin(),
                    is_positive<IndexType>());

    // replace shifted diagonals with index of diagonal in offsets array
    cusp::array1d<IndexType,cusp::host_memory> diagonal_offsets( dst.diagonal_offsets );
    for( IndexType num_diag = 0; num_diag < num_diagonals; num_diag++ )
        thrust::replace(diag_map.begin(), diag_map.end(), diagonal_offsets[num_diag], num_diag);

    // copy values to dst
    thrust::scatter(src.values.begin(), src.values.end(),
                    thrust::make_transform_iterator(
                        thrust::make_zip_iterator( thrust::make_tuple( row_indices.begin(), diag_map.begin() ) ),
                        diagonal_index_functor<IndexType>(dst.values.pitch)),
                    dst.values.values.begin());

    // shift diagonal_offsets by num_rows

    typedef typename cusp::array1d_view< thrust::constant_iterator<IndexType> > ConstantView;
    ConstantView constant_view(thrust::constant_iterator<IndexType>(dst.num_rows),
                               thrust::constant_iterator<IndexType>(dst.num_rows)+num_diagonals);
    cusp::blas::axpy(constant_view,
                     dst.diagonal_offsets,
                     IndexType(-1));

}

template <typename DerivedPolicy,
          typename SourceType,
          typename DestinationType>
void csr_to_ell(thrust::execution_policy<DerivedPolicy>& exec,
                const SourceType& src, DestinationType& dst,
                const size_t num_entries_per_row,
                const size_t alignment)
{
    typedef typename DestinationType::index_type IndexType;
    typedef typename DestinationType::value_type ValueType;
    typedef typename DestinationType::memory_space MemorySpace;

    // allocate output storage
    dst.resize(src.num_rows, src.num_cols, src.num_entries, num_entries_per_row, alignment);

    // expand row offsets into row indices
    cusp::array1d<IndexType, MemorySpace> row_indices(src.num_entries);
    cusp::detail::offsets_to_indices(src.row_offsets, row_indices);

    // compute permutation from CSR index to ELL index
    // first enumerate the entries within each row, e.g. [0, 1, 2, 0, 1, 2, 3, ...]
    cusp::array1d<IndexType, MemorySpace> permutation(src.num_entries);
    thrust::exclusive_scan_by_key(row_indices.begin(), row_indices.end(),
                                  thrust::constant_iterator<IndexType>(1),
                                  permutation.begin(),
                                  IndexType(0));

    // next, scale by pitch and add row index
    cusp::blas::axpby(permutation, row_indices,
                      permutation,
                      IndexType(dst.column_indices.pitch),
                      IndexType(1));

    // fill output with padding
    thrust::fill(dst.column_indices.values.begin(), dst.column_indices.values.end(), IndexType(-1));
    thrust::fill(dst.values.values.begin(),         dst.values.values.end(),         ValueType(0));

    // scatter CSR entries to ELL
    thrust::scatter(src.column_indices.begin(), src.column_indices.end(),
                    permutation.begin(),
                    dst.column_indices.values.begin());
    thrust::scatter(src.values.begin(), src.values.end(),
                    permutation.begin(),
                    dst.values.values.begin());
}

template <typename DerivedPolicy,
          typename SourceType,
          typename DestinationType>
void csr_to_hyb(thrust::execution_policy<DerivedPolicy>& exec,
                const SourceType& src, DestinationType& dst,
                const size_t num_entries_per_row,
                const size_t alignment)
{
    typedef typename DestinationType::index_type IndexType;
    typedef typename DestinationType::value_type ValueType;
    typedef typename DestinationType::memory_space MemorySpace;

    // expand row offsets into row indices
    cusp::array1d<IndexType, MemorySpace> row_indices(src.num_entries);
    cusp::detail::offsets_to_indices(src.row_offsets, row_indices);

    // TODO call coo_to_hyb with a coo_matrix_view

    cusp::array1d<IndexType, MemorySpace> indices(src.num_entries);
    thrust::exclusive_scan_by_key(row_indices.begin(), row_indices.end(),
                                  thrust::constant_iterator<IndexType>(1),
                                  indices.begin(),
                                  IndexType(0));

    size_t num_coo_entries = thrust::count_if(indices.begin(), indices.end(), greater_than_or_equal_to<size_t>(num_entries_per_row));
    size_t num_ell_entries = src.num_entries - num_coo_entries;

    // allocate output storage
    dst.resize(src.num_rows, src.num_cols, num_ell_entries, num_coo_entries, num_entries_per_row, alignment);

    // fill output with padding
    thrust::fill(dst.ell.column_indices.values.begin(), dst.ell.column_indices.values.end(), IndexType(-1));
    thrust::fill(dst.ell.values.values.begin(),         dst.ell.values.values.end(),         ValueType(0));

    thrust::copy_if(thrust::make_zip_iterator( thrust::make_tuple( row_indices.begin(), src.column_indices.begin(), src.values.begin() ) ),
                    thrust::make_zip_iterator( thrust::make_tuple( row_indices.end()  , src.column_indices.end()  , src.values.end()   ) ),
                    indices.begin(),
                    thrust::make_zip_iterator( thrust::make_tuple( dst.coo.row_indices.begin(), dst.coo.column_indices.begin(), dst.coo.values.begin() ) ),
                    greater_than_or_equal_to<size_t>(num_entries_per_row) );

    // next, scale by pitch and add row index
    cusp::blas::axpby(indices, row_indices,
                      indices,
                      IndexType(dst.ell.column_indices.pitch),
                      IndexType(1));

    // scatter CSR entries to ELL
    thrust::scatter_if(src.column_indices.begin(), src.column_indices.end(),
                       indices.begin(),
                       indices.begin(),
                       dst.ell.column_indices.values.begin(),
                       less_than<size_t>(dst.ell.column_indices.values.size()));
    thrust::scatter_if(src.values.begin(), src.values.end(),
                       indices.begin(),
                       indices.begin(),
                       dst.ell.values.values.begin(),
                       less_than<size_t>(dst.ell.values.values.size()));
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp

