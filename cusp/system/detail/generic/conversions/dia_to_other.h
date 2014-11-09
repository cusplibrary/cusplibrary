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
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/sort.h>

#include <cusp/blas/blas.h>
#include <cusp/detail/format_utils.h>
#include <cusp/detail/device/conversion_utils.h>
#include <cusp/detail/host/convert.h>
#include <cusp/detail/functional.h>

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

using namespace cusp::detail;

template <typename IndexType>
struct is_valid_ell_index
{
    const IndexType num_rows;

    is_valid_ell_index(const IndexType num_rows)
        : num_rows(num_rows) {}

    template <typename Tuple>
    __host__ __device__
    bool operator()(const Tuple& t) const
    {
        const IndexType i = thrust::get<0>(t);
        const IndexType j = thrust::get<1>(t);

        return i < num_rows && j != IndexType(-1);
    }
};

template <typename IndexType, typename ValueType>
struct is_valid_coo_index
{
    const IndexType num_rows;
    const IndexType num_cols;

    is_valid_coo_index(const IndexType num_rows, const IndexType num_cols)
        : num_rows(num_rows), num_cols(num_cols) {}

    template <typename Tuple>
    __host__ __device__
    bool operator()(const Tuple& t) const
    {
        const IndexType i = thrust::get<0>(t);
        const IndexType j = thrust::get<1>(t);
        const ValueType value = thrust::get<2>(t);

        return ( i > IndexType(-1) && i < num_rows ) &&
               ( j > IndexType(-1) && j < num_cols ) &&
               ( value != ValueType(0) ) ;
    }
};

template <typename DerivedPolicy,
          typename SourceType,
          typename DestinationType>
void dia_to_coo(thrust::execution_policy<DerivedPolicy>& exec,
                const SourceType& src, DestinationType& dst)
{
    typedef typename DestinationType::index_type IndexType;
    typedef typename DestinationType::value_type ValueType;
    typedef typename DestinationType::memory_space MemorySpace;

    // allocate output storage
    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    if( src.num_entries == 0 ) return;

    const IndexType pitch = src.values.pitch;
    const size_t num_entries   = src.values.values.size();
    const size_t num_diagonals = src.diagonal_offsets.size();

    // define types used to programatically generate row_indices
    typedef typename thrust::counting_iterator<IndexType> IndexIterator;
    typedef typename thrust::transform_iterator<modulus_value<IndexType>, IndexIterator> RowIndexIterator;

    RowIndexIterator row_indices_begin(IndexIterator(0), modulus_value<IndexType>(pitch));

    // define types used to programatically generate column_indices
    typedef typename thrust::device_vector<IndexType>::const_iterator ConstElementIterator;
    typedef typename thrust::transform_iterator<divide_value<IndexType>, IndexIterator> DivideIterator;
    typedef typename thrust::permutation_iterator<ConstElementIterator,DivideIterator> OffsetsPermIterator;
    typedef typename thrust::tuple<OffsetsPermIterator, RowIndexIterator> IteratorTuple;
    typedef typename thrust::zip_iterator<IteratorTuple> ZipIterator;
    typedef typename thrust::transform_iterator<sum_tuple_functor<IndexType>, ZipIterator> ColumnIndexIterator;

    DivideIterator gather_indices_begin(IndexIterator(0), divide_value<IndexType>(pitch));
    OffsetsPermIterator offsets_begin(src.diagonal_offsets.begin(), gather_indices_begin);
    ZipIterator offset_modulus_tuple(thrust::make_tuple(offsets_begin, row_indices_begin));
    ColumnIndexIterator column_indices_begin(offset_modulus_tuple, sum_tuple_functor<IndexType>());

    // copy valid entries to COO format
    //thrust::copy_if
    //  (thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices_begin, src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_diagonals))),
    //   thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices_begin, src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_diagonals))) + num_entries,
    //   thrust::make_zip_iterator(thrust::make_tuple(dst.row_indices.begin(), dst.column_indices.begin(), dst.values.begin())),
    //   is_valid_coo_index<IndexType,ValueType>(src.num_rows,src.num_cols));
    {
        // TODO remove this when copy_if can accept more parameters
        cusp::array1d<IndexType, MemorySpace> temp0(num_entries);
        cusp::array1d<IndexType, MemorySpace> temp1(num_entries);
        cusp::array1d<ValueType, MemorySpace> temp2(num_entries);
        thrust::copy(thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices_begin, src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_diagonals))),
                     thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices_begin, src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_diagonals))) + num_entries,
                     thrust::make_zip_iterator(thrust::make_tuple(temp0.begin(), temp1.begin(), temp2.begin())));
        thrust::copy_if
        (thrust::make_zip_iterator(thrust::make_tuple(temp0.begin(), temp1.begin(), temp2.begin())),
         thrust::make_zip_iterator(thrust::make_tuple(temp0.begin(), temp1.begin(), temp2.begin())) + num_entries,
         thrust::make_zip_iterator(thrust::make_tuple(dst.row_indices.begin(), dst.column_indices.begin(), dst.values.begin())),
         is_valid_coo_index<IndexType,ValueType>(src.num_rows,src.num_cols));
    }
}

template <typename DerivedPolicy,
          typename SourceType,
          typename DestinationType>
void dia_to_csr(thrust::execution_policy<DerivedPolicy>& exec,
                const SourceType& src, DestinationType& dst)
{
    typedef typename DestinationType::index_type IndexType;
    typedef typename DestinationType::value_type ValueType;
    typedef typename DestinationType::memory_space MemorySpace;

    // allocate output storage
    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    if( src.num_entries == 0 ) return;

    const IndexType pitch = src.values.pitch;
    const size_t num_entries   = src.values.values.size();
    const size_t num_diagonals = src.diagonal_offsets.size();

    // define types used to programatically generate row_indices
    typedef typename thrust::counting_iterator<IndexType> IndexIterator;
    typedef typename thrust::transform_iterator<modulus_value<IndexType>, IndexIterator> RowIndexIterator;

    RowIndexIterator row_indices_begin(IndexIterator(0), modulus_value<IndexType>(pitch));

    // define types used to programatically generate column_indices
    // typedef typename thrust::device_vector<IndexType>::const_iterator                      ConstElementIterator;
    typedef typename SourceType::diagonal_offsets_array_type::const_iterator               ConstElementIterator;
    typedef typename thrust::transform_iterator<divide_value<IndexType>, IndexIterator>    DivideIterator;
    typedef typename thrust::permutation_iterator<ConstElementIterator,DivideIterator>     OffsetsPermIterator;
    typedef typename thrust::tuple<OffsetsPermIterator, RowIndexIterator>                  IteratorTuple;
    typedef typename thrust::zip_iterator<IteratorTuple>                                   ZipIterator;
    typedef typename thrust::transform_iterator<sum_tuple_functor<IndexType>, ZipIterator> ColumnIndexIterator;

    DivideIterator gather_indices_begin(IndexIterator(0), divide_value<IndexType>(pitch));
    OffsetsPermIterator offsets_begin(src.diagonal_offsets.begin(), gather_indices_begin);
    ZipIterator offset_modulus_tuple(thrust::make_tuple(offsets_begin, row_indices_begin));
    ColumnIndexIterator column_indices_begin(offset_modulus_tuple, sum_tuple_functor<IndexType>());

    cusp::array1d<IndexType, MemorySpace> row_indices(src.num_entries);

    // copy valid entries to COO format
    {
        // TODO remove this when copy_if can accept more parameters
        cusp::array1d<IndexType, MemorySpace> temp0(num_entries);
        cusp::array1d<IndexType, MemorySpace> temp1(num_entries);
        cusp::array1d<ValueType, MemorySpace> temp2(num_entries);
        thrust::copy(thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices_begin, src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType,cusp::column_major,cusp::row_major>(src.values.num_rows, src.values.num_cols, src.values.pitch))),
                     thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices_begin, src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType,cusp::column_major,cusp::row_major>(src.values.num_rows, src.values.num_cols, src.values.pitch))) + num_entries,
                     thrust::make_zip_iterator(thrust::make_tuple(temp0.begin(), temp1.begin(), temp2.begin())));
        thrust::copy_if
        (thrust::make_zip_iterator(thrust::make_tuple(temp0.begin(), temp1.begin(), temp2.begin())),
         thrust::make_zip_iterator(thrust::make_tuple(temp0.begin(), temp1.begin(), temp2.begin())) + num_entries,
         thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), dst.column_indices.begin(), dst.values.begin())),
         is_valid_coo_index<IndexType,ValueType>(src.num_rows,src.num_cols));
    }
    //thrust::copy_if
    //  (thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices_begin, src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_diagonals))),
    //   thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices_begin, src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_diagonals))) + num_entries,
    //   thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), dst.column_indices.begin(), dst.values.begin())),
    //   is_valid_coo_index<IndexType,ValueType>(src.num_rows,src.num_cols));

    cusp::detail::indices_to_offsets( row_indices, dst.row_offsets );
}


} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp
