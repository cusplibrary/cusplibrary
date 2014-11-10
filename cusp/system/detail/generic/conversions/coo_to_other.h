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

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
typename enable_if_same_system<SourceType,DestinationType>::type
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::coo_format&,
        cusp::csr_format&)
{
    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    cusp::detail::indices_to_offsets(src.row_indices, dst.row_offsets);
    cusp::copy(src.column_indices, dst.column_indices);
    cusp::copy(src.values,         dst.values);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
typename enable_if_same_system<SourceType,DestinationType>::type
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::coo_format&,
        cusp::dia_format&,
        size_t alignment)
{
    typedef typename DestinationType::index_type IndexType;
    typedef typename DestinationType::value_type ValueType;
    typedef typename DestinationType::memory_space MemorySpace;

    // compute number of occupied diagonals and enumerate them
    cusp::array1d<IndexType,MemorySpace> diag_map(src.num_entries);
    thrust::transform(thrust::make_zip_iterator( thrust::make_tuple( src.row_indices.begin(), src.column_indices.begin() ) ),
                      thrust::make_zip_iterator( thrust::make_tuple( src.row_indices.end()  , src.column_indices.end() ) )  ,
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
                        thrust::make_zip_iterator( thrust::make_tuple( src.row_indices.begin(), diag_map.begin() ) ),
                        diagonal_index_functor<IndexType>(dst.values.pitch)),
                    dst.values.values.begin());

    thrust::transform(dst.diagonal_offsets.begin(), dst.diagonal_offsets.end(),
                      thrust::constant_iterator<IndexType>(dst.num_rows),
                      dst.diagonal_offsets.begin(), thrust::minus<IndexType>());
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
typename enable_if_same_system<SourceType,DestinationType>::type
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::coo_format&,
        cusp::ell_format&,
        size_t num_entries_per_row,
        size_t alignment)
{
    typedef typename DestinationType::index_type IndexType;
    typedef typename DestinationType::value_type ValueType;
    typedef typename DestinationType::memory_space MemorySpace;

    if (src.num_entries == 0)
    {
        dst.resize(src.num_rows, src.num_cols, src.num_entries, 0);
        return;
    }

    // allocate output storage
    dst.resize(src.num_rows, src.num_cols, src.num_entries, num_entries_per_row, alignment);

    // compute permutation from COO index to ELL index
    // first enumerate the entries within each row, e.g. [0, 1, 2, 0, 1, 2, 3, ...]
    cusp::array1d<IndexType, MemorySpace> permutation(src.num_entries);

    thrust::exclusive_scan_by_key(src.row_indices.begin(), src.row_indices.end(),
                                  thrust::constant_iterator<IndexType>(1),
                                  permutation.begin(),
                                  IndexType(0));

    // next, scale by pitch and add row index
    cusp::blas::axpby(permutation, src.row_indices,
                      permutation,
                      IndexType(dst.column_indices.pitch),
                      IndexType(1));

    // fill output with padding
    thrust::fill(dst.column_indices.values.begin(), dst.column_indices.values.end(), IndexType(-1));
    thrust::fill(dst.values.values.begin(),         dst.values.values.end(),         ValueType(0));

    // scatter COO entries to ELL
    thrust::scatter(src.column_indices.begin(), src.column_indices.end(),
                    permutation.begin(),
                    dst.column_indices.values.begin());
    thrust::scatter(src.values.begin(), src.values.end(),
                    permutation.begin(),
                    dst.values.values.begin());
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
typename enable_if_same_system<SourceType,DestinationType>::type
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::coo_format& format1,
        cusp::hyb_format& format2,
        size_t num_entries_per_row,
        size_t alignment)
{
    typedef typename DestinationType::index_type IndexType;
    typedef typename DestinationType::value_type ValueType;
    typedef typename DestinationType::memory_space MemorySpace;

    cusp::array1d<IndexType,MemorySpace> indices(src.num_entries);
    thrust::exclusive_scan_by_key(src.row_indices.begin(), src.row_indices.end(),
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

    // write tail of each row to COO portion
    thrust::copy_if
    (thrust::make_zip_iterator( thrust::make_tuple( src.row_indices.begin(), src.column_indices.begin(), src.values.begin() ) ),
     thrust::make_zip_iterator( thrust::make_tuple( src.row_indices.end()  , src.column_indices.end()  , src.values.end()   ) ),
     indices.begin(),
     thrust::make_zip_iterator( thrust::make_tuple( dst.coo.row_indices.begin(), dst.coo.column_indices.begin(), dst.coo.values.begin() ) ),
     greater_than_or_equal_to<size_t>(num_entries_per_row) );

    assert(dst.ell.column_indices.pitch == dst.ell.values.pitch);

    size_t pitch = dst.ell.column_indices.pitch;

    // next, scale by pitch and add row index
    cusp::blas::axpby(indices, src.row_indices,
                      indices,
                      IndexType(pitch),
                      IndexType(1));

    // scatter COO entries to ELL
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
//// fused version appears to be slightly slower
//  thrust::scatter_if(thrust::make_zip_iterator(thrust::make_tuple(src.column_indices.begin(), src.values.begin())),
//                     thrust::make_zip_iterator(thrust::make_tuple(src.column_indices.end(),   src.values.end())),
//                     indices.begin(),
//                     indices.begin(),
//                     thrust::make_zip_iterator(thrust::make_tuple(dst.ell.column_indices.values.begin(), dst.ell.values.values.begin())),
//                     less_than<size_t>(dst.ell.column_indices.values.size()));
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp

