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

#include <cusp/convert.h>
#include <cusp/ell_matrix.h>
#include <cusp/coo_matrix.h>

namespace cusp
{

//////////////////
// Constructors //
//////////////////

// construct from another matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
hyb_matrix<IndexType,ValueType,MemorySpace>
::hyb_matrix(const MatrixType& matrix)
{
    cusp::convert(matrix, *this);
}

//////////////////////
// Member Functions //
//////////////////////

template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
hyb_matrix<IndexType,ValueType,MemorySpace>&
hyb_matrix<IndexType,ValueType,MemorySpace>
::operator=(const MatrixType& matrix)
{
    cusp::convert(matrix, *this);

    return *this;
}

template <typename IndexType, typename ValueType, class MemorySpace>
typename hyb_matrix<IndexType,ValueType,MemorySpace>::coo_view_type
hyb_matrix<IndexType,ValueType,MemorySpace>
::ascoo(void)
{
    typedef thrust::counting_iterator<IndexType>                                                            CountingIterator;
    typedef thrust::transform_iterator<cusp::detail::divide_value<IndexType>, CountingIterator>             RowIndexIterator;
    typedef typename ell_matrix_type::column_indices_array_type::values_array_type::iterator                ColumnIndexIterator;
    typedef typename ell_matrix_type::values_array_type::values_array_type::iterator                        ValueIterator;

    typedef cusp::detail::logical_to_other_physical_functor<IndexType, cusp::row_major, cusp::column_major> PermFunctor;
    typedef thrust::transform_iterator<PermFunctor, CountingIterator>                                       PermIndexIterator;
    typedef thrust::permutation_iterator<ColumnIndexIterator, PermIndexIterator>                            PermColumnIndexIterator;
    typedef thrust::permutation_iterator<ValueIterator, PermIndexIterator>                                  PermValueIterator;

    typedef typename cusp::array1d<IndexType,MemorySpace>::iterator                                         IndexIterator;
    typedef cusp::join_iterator<RowIndexIterator,ColumnIndexIterator,IndexIterator>                         JoinRowIterator;
    typedef cusp::join_iterator<PermColumnIndexIterator,ColumnIndexIterator,IndexIterator>                  JoinColumnIterator;
    typedef cusp::join_iterator<PermValueIterator,ValueIterator,IndexIterator>                              JoinValueIterator;

    typedef typename coo_view_type::row_indices_array_type                                                  Array1;
    typedef typename coo_view_type::column_indices_array_type                                               Array2;
    typedef typename coo_view_type::values_array_type                                                       Array3;

    const int    X               = ell_matrix_type::invalid_index;
    const size_t ell_num_entries = ell.column_indices.num_entries;
    const size_t coo_num_entries = coo.num_entries;
    const size_t total           = ell_num_entries + coo_num_entries;

    indices.resize(total);

    RowIndexIterator        row_indices_begin(CountingIterator(0), cusp::detail::divide_value<IndexType>(ell.values.num_cols));
    PermIndexIterator       perm_indices_begin(CountingIterator(0), PermFunctor(ell.values.num_rows, ell.values.num_cols, ell.values.pitch));
    PermColumnIndexIterator perm_column_indices_begin(ell.column_indices.values.begin(), perm_indices_begin);
    PermValueIterator       perm_values_begin(ell.values.values.begin(), perm_indices_begin);

    // TODO : Remove this WAR when Thrust v1.9 is released, related issue #635
    cusp::array1d<IndexType,MemorySpace> row_indices(row_indices_begin, row_indices_begin + ell_num_entries);

    thrust::merge_by_key(thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), ell.column_indices.values.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), ell.column_indices.values.begin())) + ell_num_entries,
                         thrust::make_zip_iterator(thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(coo.row_indices.begin(), coo.column_indices.begin())) + coo_num_entries,
                         thrust::counting_iterator<IndexType>(0),
                         thrust::counting_iterator<IndexType>(ell_num_entries),
                         thrust::make_discard_iterator(),
                         indices.begin(),
                         cusp::detail::coo_tuple_comp<IndexType>());

    JoinRowIterator    rows(row_indices_begin, row_indices_begin + ell_num_entries,
                            coo.row_indices.begin(), coo.row_indices.end(),
                            indices.begin());
    JoinColumnIterator cols(perm_column_indices_begin, perm_column_indices_begin + ell_num_entries,
                            coo.column_indices.begin(), coo.column_indices.end(),
                            indices.begin());
    JoinValueIterator  vals(perm_values_begin, perm_values_begin + ell_num_entries,
                            coo.values.begin(), coo.values.end(),
                            indices.begin());

    int num_invalid = indices.end() - thrust::remove_if(indices.begin(), indices.end(), cols.begin(), thrust::placeholders::_1 == X);
    indices.resize(total - num_invalid);

    Array1 rows_array(rows.begin(), rows.end()-num_invalid);
    Array2 cols_array(cols.begin(), cols.end()-num_invalid);
    Array3 vals_array(vals.begin(), vals.end()-num_invalid);

    return coo_view_type(Parent::num_rows, Parent::num_cols, Parent::num_entries, rows_array, cols_array, vals_array);
}

template <typename Matrix1, typename Matrix2, typename IndexType, typename ValueType, typename MemorySpace>
void
hyb_matrix_view<Matrix1,Matrix2,IndexType,ValueType,MemorySpace>
::resize(size_t num_rows, size_t num_cols,
         size_t num_ell_entries, size_t num_coo_entries,
         size_t num_entries_per_row, size_t alignment)
{
    Parent::resize(num_rows, num_cols, num_ell_entries + num_coo_entries);
    ell.resize(num_rows, num_cols, num_ell_entries, num_entries_per_row, alignment);
    coo.resize(num_rows, num_cols, num_coo_entries);
}

} // end namespace cusp

