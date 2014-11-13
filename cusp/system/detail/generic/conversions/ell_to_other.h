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
             cusp::ell_format&,
             cusp::coo_format&)
{
    typedef typename DestinationType::index_type IndexType;

    const IndexType pitch = src.column_indices.pitch;

    // define types used to programatically generate row_indices
    typedef typename thrust::counting_iterator<IndexType> IndexIterator;
    typedef typename thrust::transform_iterator<modulus_value<IndexType>, IndexIterator> RowIndexIterator;

    RowIndexIterator row_indices_begin(IndexIterator(0), modulus_value<IndexType>(pitch));

    // compute true number of nonzeros in ELL
    const IndexType num_entries =
        thrust::count_if
        (thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin())),
         thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin())) + src.column_indices.values.size(),
         is_valid_ell_index<IndexType>(src.num_rows));

    // allocate output storage
    dst.resize(src.num_rows, src.num_cols, num_entries);

    // copy valid entries to COO format
    thrust::copy_if
    (thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin(), src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), logical_to_other_physical_functor<IndexType,cusp::row_major,cusp::column_major>(src.values.num_rows, src.values.num_cols, src.values.pitch))),
     thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin(), src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), logical_to_other_physical_functor<IndexType,cusp::row_major,cusp::column_major>(src.values.num_rows, src.values.num_cols, src.values.pitch))) + src.column_indices.values.size(),
     thrust::make_zip_iterator(thrust::make_tuple(dst.row_indices.begin(), dst.column_indices.begin(), dst.values.begin())),
     is_valid_ell_index<IndexType>(src.num_rows));
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
typename enable_if_same_system<SourceType,DestinationType>::type
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::ell_format&,
        cusp::csr_format&)
{
    typedef typename DestinationType::index_type IndexType;
    typedef typename DestinationType::memory_space MemorySpace;

    const IndexType pitch               = src.column_indices.pitch;
    const IndexType num_entries_per_row = src.column_indices.num_cols;

    // define types used to programatically generate row_indices
    typedef typename thrust::counting_iterator<IndexType> IndexIterator;
    typedef typename thrust::transform_iterator<modulus_value<IndexType>, IndexIterator> RowIndexIterator;

    RowIndexIterator row_indices_begin(IndexIterator(0), modulus_value<IndexType>(pitch));

    // compute true number of nonzeros in ELL
    const IndexType num_entries =
        thrust::count_if
        (thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin())),
         thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin())) + src.column_indices.values.size(),
         is_valid_ell_index<IndexType>(src.num_rows));

    // allocate output storage
    dst.resize(src.num_rows, src.num_cols, num_entries);

    // create temporary row_indices array to capture valid ELL row indices
    cusp::array1d<IndexType, MemorySpace> row_indices(num_entries);

    // copy valid entries to mixed COO/CSR format
    thrust::copy_if
    (thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin(), src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), logical_to_other_physical_functor<IndexType,cusp::row_major,cusp::column_major>(src.values.num_rows, src.values.num_cols, src.values.pitch))),
     thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin(), src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), logical_to_other_physical_functor<IndexType,cusp::row_major,cusp::column_major>(src.values.num_rows, src.values.num_cols, src.values.pitch))) + src.column_indices.values.size(),
     thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), dst.column_indices.begin(), dst.values.begin())),
     is_valid_ell_index<IndexType>(src.num_rows));

    // convert COO row_indices to CSR row_offsets
    cusp::detail::indices_to_offsets(row_indices, dst.row_offsets);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
typename enable_if_same_system<SourceType,DestinationType>::type
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::ell_format&,
        cusp::dia_format&,
        size_t alignment)
{
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
typename enable_if_same_system<SourceType,DestinationType>::type
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::ell_format&,
        cusp::hyb_format&)
{
    // just copy into ell part of destination
    dst.resize(src.num_rows, src.num_cols,
               src.num_entries, 0,
               src.column_indices.num_cols);

    cusp::copy(exec, src, dst.ell);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp
