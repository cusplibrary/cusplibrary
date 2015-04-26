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
#include <thrust/reduce.h>

#include <thrust/system/detail/generic/tag.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cusp/detail/format.h>
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
              const LinearOperator&  A,
              const MatrixOrVector1& B,
              MatrixOrVector2& C,
              UnaryFunction    initialize,
              BinaryFunction1  combine,
              BinaryFunction2  reduce,
              dia_format,
              array1d_format,
              array1d_format)
{
    typedef typename LinearOperator::index_type IndexType;
    typedef typename LinearOperator::value_type ValueType;
    typedef typename LinearOperator::memory_space MemorySpace;

    // define types used to programatically generate row_indices
    typedef typename thrust::counting_iterator<IndexType>                                        IndexIterator;
    typedef typename thrust::transform_iterator<divide_value<IndexType>, IndexIterator>          RowIndexIterator;

    // define types used to programatically generate column_indices
    typedef typename cusp::array1d<IndexType,MemorySpace>::const_iterator                        ConstElementIterator;
    typedef typename thrust::transform_iterator<modulus_value<IndexType>, IndexIterator>         ModulusIterator;
    typedef typename thrust::permutation_iterator<ConstElementIterator,ModulusIterator>          OffsetsPermIterator;
    typedef typename thrust::tuple<OffsetsPermIterator, RowIndexIterator>                        IteratorTuple;
    typedef typename thrust::zip_iterator<IteratorTuple>                                         ZipIterator;
    typedef typename thrust::transform_iterator<sum_tuple_functor<IndexType>, ZipIterator>       ColumnIndexIterator;

    typedef logical_to_other_physical_functor<IndexType, cusp::row_major, cusp::column_major>    PermFunctor;
    typedef typename LinearOperator::values_array_type::values_array_type::const_iterator        ValueIterator;
    typedef typename thrust::transform_iterator<PermFunctor, IndexIterator>                      PermIndexIterator;
    typedef typename thrust::permutation_iterator<ValueIterator, PermIndexIterator>              PermValueIterator;

    if(A.num_entries == 0)
    {
        thrust::transform(exec, C.begin(), C.end(), C.begin(), initialize);
        return;
    }

    RowIndexIterator row_indices_begin(IndexIterator(0), divide_value<IndexType>(A.values.num_cols));

    ModulusIterator gather_indices_begin(IndexIterator(0), modulus_value<IndexType>(A.values.num_cols));
    OffsetsPermIterator offsets_begin(A.diagonal_offsets.begin(), gather_indices_begin);
    ZipIterator offset_modulus_tuple(thrust::make_tuple(offsets_begin, row_indices_begin));
    ColumnIndexIterator column_indices_begin(offset_modulus_tuple, sum_tuple_functor<IndexType>());

    PermIndexIterator   perm_indices_begin(IndexIterator(0),   PermFunctor(A.values.num_rows, A.values.num_cols, A.values.pitch));
    PermValueIterator   perm_values_begin(A.values.values.begin(),  perm_indices_begin);

    thrust::detail::temporary_array<ValueType, DerivedPolicy> vals(exec, A.num_rows);

    thrust::reduce_by_key(exec,
                          row_indices_begin, row_indices_begin + A.values.values.size(),
                          thrust::make_transform_iterator(
                              thrust::make_zip_iterator(
                                  thrust::make_tuple(
                                    thrust::make_permutation_iterator(B.begin(),
                                      thrust::make_transform_iterator(column_indices_begin,
                                        valid_index_functor<IndexType>(A.num_cols))),
                                      perm_values_begin)),
                              combine_tuple_base_functor<BinaryFunction1>()),
                          thrust::make_discard_iterator(),
                          vals.begin(),
                          thrust::equal_to<IndexType>(),
                          reduce);

    thrust::transform(exec,
                      vals.begin(),
                      vals.end(),
                      thrust::make_transform_iterator(C.begin(), initialize),
                      C.begin(),
                      reduce);
}

template <typename DerivedPolicy,
         typename LinearOperator, typename MatrixOrVector1, typename MatrixOrVector2,
         typename UnaryFunction,  typename BinaryFunction1, typename BinaryFunction2>
void multiply(thrust::execution_policy<DerivedPolicy> &exec,
              const LinearOperator&  A,
              const MatrixOrVector1& B,
              MatrixOrVector2& C,
              UnaryFunction    initialize,
              BinaryFunction1  combine,
              BinaryFunction2  reduce,
              ell_format,
              array1d_format,
              array1d_format)
{
    typedef typename LinearOperator::index_type IndexType;
    typedef typename LinearOperator::value_type ValueType;

    typedef logical_to_other_physical_functor<IndexType,cusp::row_major,cusp::column_major> LogicalFunctor;

    // define types used to programatically generate row_indices
    typedef typename thrust::counting_iterator<IndexType> IndexIterator;
    typedef typename thrust::transform_iterator<divide_value<IndexType>, IndexIterator> RowIndexIterator;
    typedef typename thrust::transform_iterator<LogicalFunctor, IndexIterator> StrideIndexIterator;

    if(A.num_entries == 0)
    {
        thrust::transform(exec, C.begin(), C.end(), C.begin(), initialize);
        return;
    }

    size_t num_entries = A.values.num_entries;
    size_t num_entries_per_row = A.column_indices.num_cols;
    LogicalFunctor logical_functor(A.values.num_rows, A.values.num_cols, A.values.pitch);

    RowIndexIterator row_indices_begin(IndexIterator(0), divide_value<IndexType>(num_entries_per_row));
    StrideIndexIterator stride_indices_begin(IndexIterator(0), logical_functor);
    thrust::detail::temporary_array<ValueType, DerivedPolicy> vals(exec, A.num_rows);

    thrust::reduce_by_key(exec,
                          row_indices_begin, row_indices_begin + num_entries,
                          thrust::make_permutation_iterator(
                          thrust::make_transform_iterator(
                              thrust::make_zip_iterator(
                                  thrust::make_tuple(
                                    thrust::make_permutation_iterator(B.begin(),
                                      thrust::make_transform_iterator(
                                        A.column_indices.values.begin(),
                                        valid_index_functor<IndexType>(A.num_cols))),
                                      A.values.values.begin())),
                              combine_tuple_base_functor<BinaryFunction1>()),
                          stride_indices_begin),
                          thrust::make_discard_iterator(),
                          vals.begin(),
                          thrust::equal_to<IndexType>(),
                          reduce);

    thrust::transform(exec, vals.begin(), vals.end(),
                      thrust::make_transform_iterator(C.begin(), initialize),
                      C.begin(), reduce);
}

template <typename DerivedPolicy,
         typename LinearOperator, typename MatrixOrVector1, typename MatrixOrVector2,
         typename UnaryFunction,  typename BinaryFunction1, typename BinaryFunction2>
void multiply(thrust::execution_policy<DerivedPolicy> &exec,
              LinearOperator&  A,
              MatrixOrVector1& B,
              MatrixOrVector2& C,
              UnaryFunction    initialize,
              BinaryFunction1  combine,
              BinaryFunction2  reduce,
              coo_format,
              array1d_format,
              array1d_format)
{
    typedef typename LinearOperator::index_type IndexType;
    typedef typename LinearOperator::value_type ValueType;

    thrust::detail::temporary_array<IndexType, DerivedPolicy> rows(exec, A.num_rows);
    thrust::detail::temporary_array<ValueType, DerivedPolicy> vals(exec, A.num_rows);

    typename thrust::detail::temporary_array<IndexType, DerivedPolicy>::iterator rows_end;
    typename thrust::detail::temporary_array<ValueType, DerivedPolicy>::iterator vals_end;

    thrust::tie(rows_end, vals_end) =
        thrust::reduce_by_key(exec,
                              A.row_indices.begin(), A.row_indices.end(),
                              thrust::make_transform_iterator(
                                  thrust::make_zip_iterator(
                                      thrust::make_tuple(
                                        thrust::make_permutation_iterator(B.begin(),
                                          A.column_indices.begin()),
                                          A.values.begin())),
                              combine_tuple_base_functor<BinaryFunction1>()),
                              rows.begin(),
                              vals.begin(),
                              thrust::equal_to<IndexType>(),
                              reduce);

    thrust::transform(exec, C.begin(), C.end(), C.begin(), initialize);

    int num_entries = rows_end - rows.begin();
    thrust::transform(exec,
                      thrust::make_permutation_iterator(C.begin(), rows.begin()),
                      thrust::make_permutation_iterator(C.begin(), rows.begin()) + num_entries,
                      vals.begin(),
                      thrust::make_permutation_iterator(C.begin(), rows.begin()),
                      reduce);
}

template <typename DerivedPolicy,
         typename LinearOperator, typename MatrixOrVector1, typename MatrixOrVector2,
         typename UnaryFunction,  typename BinaryFunction1, typename BinaryFunction2>
void multiply(thrust::execution_policy<DerivedPolicy> &exec,
              LinearOperator&  A,
              MatrixOrVector1& B,
              MatrixOrVector2& C,
              UnaryFunction    initialize,
              BinaryFunction1  combine,
              BinaryFunction2  reduce,
              csr_format,
              array1d_format,
              array1d_format)
{
    typedef typename LinearOperator::index_type IndexType;
    typedef typename thrust::detail::temporary_array<IndexType, DerivedPolicy>::iterator TempIterator;

    typedef typename cusp::array1d_view<TempIterator>::view           RowView;
    typedef typename LinearOperator::column_indices_array_type::view  ColView;
    typedef typename LinearOperator::values_array_type::view          ValView;

    thrust::detail::temporary_array<IndexType, DerivedPolicy> temp(exec, A.num_entries);
    cusp::array1d_view<TempIterator> row_indices(temp.begin(), temp.end());
    cusp::offsets_to_indices(A.row_offsets, row_indices);

    cusp::coo_matrix_view<RowView,ColView,ValView> A_coo_view(A.num_rows, A.num_cols, A.num_entries,
                                                              cusp::make_array1d_view(row_indices),
                                                              cusp::make_array1d_view(A.column_indices),
                                                              cusp::make_array1d_view(A.values));

    cusp::multiply(exec, A_coo_view, B, C, initialize, combine, reduce);
}

template <typename DerivedPolicy,
         typename LinearOperator, typename MatrixOrVector1, typename MatrixOrVector2,
         typename UnaryFunction,  typename BinaryFunction1, typename BinaryFunction2>
void multiply(thrust::execution_policy<DerivedPolicy> &exec,
              LinearOperator&  A,
              MatrixOrVector1& B,
              MatrixOrVector2& C,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce,
              cusp::permutation_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
    // TODO : initialize, combine, and reduce are ignored
    thrust::gather(A.permutation.begin(), A.permutation.end(), B.begin(), C.begin());
}

template <typename DerivedPolicy,
          typename LinearOperator, typename MatrixOrVector1, typename MatrixOrVector2,
          typename UnaryFunction,  typename BinaryFunction1, typename BinaryFunction2>
void multiply(thrust::execution_policy<DerivedPolicy> &exec,
              LinearOperator&  A,
              MatrixOrVector1& B,
              MatrixOrVector2& C,
              UnaryFunction  initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce,
              hyb_format,
              array1d_format,
              array1d_format)
{
    typedef typename MatrixOrVector2::value_type ValueType;

    cusp::multiply(exec, A.ell, B, C, initialize, combine, reduce);
    cusp::multiply(exec, A.coo, B, C, thrust::identity<ValueType>(), combine, reduce);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp
