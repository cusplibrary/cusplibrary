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

#include <cusp/array1d.h>
#include <cusp/convert.h>
#include <cusp/format_utils.h>
#include <cusp/sort.h>

#include <cusp/iterator/join_iterator.h>

#include <cusp/detail/array2d_format_utils.h>

#include <thrust/merge.h>
#include <thrust/remove.h>
#include <thrust/sort.h>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

namespace cusp
{

//////////////////
// Constructors //
//////////////////

// construct from a different matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
coo_matrix<IndexType,ValueType,MemorySpace>
::coo_matrix(const MatrixType& matrix)
{
    cusp::convert(matrix, *this);
}

////////////////////////////////
// Container Member Functions //
////////////////////////////////

template <typename IndexType, typename ValueType, class MemorySpace>
void
coo_matrix<IndexType,ValueType,MemorySpace>
::resize(const size_t num_rows, const size_t num_cols, const size_t num_entries)
{
    Parent::resize(num_rows, num_cols, num_entries);
    row_indices.resize(num_entries);
    column_indices.resize(num_entries);
    values.resize(num_entries);
}

template <typename IndexType, typename ValueType, class MemorySpace>
void
coo_matrix<IndexType,ValueType,MemorySpace>
::swap(coo_matrix& matrix)
{
    Parent::swap(matrix);
    row_indices.swap(matrix.row_indices);
    column_indices.swap(matrix.column_indices);
    values.swap(matrix.values);
}

// assignment from another matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
coo_matrix<IndexType,ValueType,MemorySpace>&
coo_matrix<IndexType,ValueType,MemorySpace>
::operator=(const MatrixType& matrix)
{
    cusp::convert(matrix, *this);

    return *this;
}

// sort matrix elements by row index
template <typename IndexType, typename ValueType, class MemorySpace>
void
coo_matrix<IndexType,ValueType,MemorySpace>
::sort_by_row(void)
{
    cusp::sort_by_row(row_indices, column_indices, values);
}

// sort matrix elements by row index
template <typename IndexType, typename ValueType, class MemorySpace>
void
coo_matrix<IndexType,ValueType,MemorySpace>
::sort_by_row_and_column(void)
{
    cusp::sort_by_row_and_column(row_indices, column_indices, values);
}

// determine whether matrix elements are sorted by row index
template <typename IndexType, typename ValueType, class MemorySpace>
bool
coo_matrix<IndexType,ValueType,MemorySpace>
::is_sorted_by_row(void)
{
    return thrust::is_sorted(row_indices.begin(), row_indices.end());
}

// determine whether matrix elements are sorted by row and column index
template <typename IndexType, typename ValueType, class MemorySpace>
bool
coo_matrix<IndexType,ValueType,MemorySpace>
::is_sorted_by_row_and_column(void)
{
    return thrust::is_sorted
           (thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), column_indices.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(row_indices.end(),   column_indices.end())));
}

//////////////////
// Constructors //
//////////////////

template <typename Array1, typename Array2, typename Array3, typename IndexType, typename ValueType, typename MemorySpace>
template <typename MatrixType>
coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
::coo_matrix_view(const MatrixType& matrix,
                  typename thrust::detail::enable_if_convertible<typename MatrixType::format,hyb_format>::type*)
 : Parent(matrix)
{
    using namespace cusp::detail;

    typedef typename MatrixType::ell_matrix_type                                                ell_matrix_type;
    typedef thrust::counting_iterator<IndexType>                                                CountingIterator;
    typedef thrust::transform_iterator<divide_value<IndexType>, CountingIterator>               RowIndexIterator;
    typedef typename ell_matrix_type::column_indices_array_type::values_array_type::const_iterator    ColumnIndexIterator;
    typedef typename ell_matrix_type::values_array_type::values_array_type::const_iterator            ValueIterator;

    typedef logical_to_other_physical_functor<IndexType, cusp::row_major, cusp::column_major>   PermFunctor;
    typedef thrust::transform_iterator<PermFunctor, CountingIterator>                           PermIndexIterator;
    typedef thrust::permutation_iterator<ColumnIndexIterator, PermIndexIterator>                PermColumnIndexIterator;
    typedef thrust::permutation_iterator<ValueIterator, PermIndexIterator>                      PermValueIterator;

    typedef typename cusp::array1d<IndexType,MemorySpace>::const_iterator                       IndexIterator;
    typedef cusp::join_iterator<RowIndexIterator,ColumnIndexIterator,IndexIterator>             JoinRowIterator;
    typedef cusp::join_iterator<PermColumnIndexIterator,ColumnIndexIterator,IndexIterator>      JoinColumnIterator;
    typedef cusp::join_iterator<PermValueIterator,ValueIterator,IndexIterator>                  JoinValueIterator;

    const int    X               = ell_matrix_type::invalid_index;
    const size_t ell_num_entries = matrix.ell.column_indices.num_entries;
    const size_t coo_num_entries = matrix.coo.num_entries;
    const size_t total           = ell_num_entries + coo_num_entries;
    const size_t num_invalid = ell_num_entries - matrix.ell.num_entries;

    RowIndexIterator        row_indices_begin(CountingIterator(0), divide_value<IndexType>(matrix.ell.values.num_cols));
    PermIndexIterator       perm_indices_begin(CountingIterator(0), PermFunctor(matrix.ell.values.num_rows, matrix.ell.values.num_cols, matrix.ell.values.pitch));
    PermColumnIndexIterator perm_column_indices_begin(matrix.ell.column_indices.values.begin(), perm_indices_begin);
    PermValueIterator       perm_values_begin(matrix.ell.values.values.begin(), perm_indices_begin);

    indices.resize(total);

    // TODO : Remove this WAR when Thrust v1.9 is released, related issue #635
    {
      cusp::array1d<IndexType,MemorySpace> temp_row_indices(row_indices_begin, row_indices_begin + ell_num_entries);
      cusp::array1d<IndexType,MemorySpace> temp_column_indices(perm_column_indices_begin, perm_column_indices_begin + ell_num_entries);

      thrust::merge_by_key(thrust::make_zip_iterator(thrust::make_tuple(temp_row_indices.begin(), temp_column_indices.begin())),
                           thrust::make_zip_iterator(thrust::make_tuple(temp_row_indices.begin(), temp_column_indices.begin())) + ell_num_entries,
                           thrust::make_zip_iterator(thrust::make_tuple(matrix.coo.row_indices.begin(), matrix.coo.column_indices.begin())),
                           thrust::make_zip_iterator(thrust::make_tuple(matrix.coo.row_indices.begin(), matrix.coo.column_indices.begin())) + coo_num_entries,
                           thrust::counting_iterator<IndexType>(0),
                           thrust::counting_iterator<IndexType>(ell_num_entries),
                           thrust::make_discard_iterator(),
                           indices.begin(),
                           coo_tuple_comp<IndexType>());
    }

    JoinRowIterator    rows_iter(row_indices_begin, row_indices_begin + ell_num_entries,
                                 matrix.coo.row_indices.begin(), matrix.coo.row_indices.end(),
                                 indices.begin());
    JoinColumnIterator cols_iter(perm_column_indices_begin, perm_column_indices_begin + ell_num_entries,
                                 matrix.coo.column_indices.begin(), matrix.coo.column_indices.end(),
                                 indices.begin());
    JoinValueIterator  vals_iter(perm_values_begin, perm_values_begin + ell_num_entries,
                                 matrix.coo.values.begin(), matrix.coo.values.end(),
                                 indices.begin());

    cusp::array1d<IndexType,MemorySpace> temp_indices(indices);
    thrust::remove_if(temp_indices.begin(), temp_indices.end(), cols_iter.begin(), thrust::placeholders::_1 == X);
    thrust::copy(temp_indices.begin(), temp_indices.begin() + total - num_invalid, indices.begin());

    row_indices_array_type    rows_array(rows_iter.begin(), rows_iter.end()-num_invalid);
    column_indices_array_type cols_array(cols_iter.begin(), cols_iter.end()-num_invalid);
    values_array_type         vals_array(vals_iter.begin(), vals_iter.end()-num_invalid);

    row_indices    = rows_array;
    column_indices = cols_array;
    values         = vals_array;
}

template <typename Array1, typename Array2, typename Array3, typename IndexType, typename ValueType, typename MemorySpace>
template <typename MatrixType>
coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
::coo_matrix_view(const MatrixType& matrix,
                  typename thrust::detail::enable_if_convertible<typename MatrixType::format,csr_format>::type*)
 : Parent(matrix)
{
    indices.resize(matrix.num_entries);
    cusp::offsets_to_indices(matrix.row_offsets, indices);

    row_indices    = row_indices_array_type(indices.begin(), indices.end());
    column_indices = column_indices_array_type(matrix.column_indices.begin(), matrix.column_indices.end());
    values         = values_array_type(matrix.values.begin(), matrix.values.end());
}

///////////////////////////
// View Member Functions //
///////////////////////////

template <typename Array1, typename Array2, typename Array3, typename IndexType, typename ValueType, typename MemorySpace>
void
coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
::resize(const size_t num_rows, const size_t num_cols, const size_t num_entries)
{
    Parent::resize(num_rows, num_cols, num_entries);
    row_indices.resize(num_entries);
    column_indices.resize(num_entries);
    values.resize(num_entries);
}

// sort matrix elements by row index
template <typename Array1, typename Array2, typename Array3, typename IndexType, typename ValueType, typename MemorySpace>
void
coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
::sort_by_row(void)
{
    cusp::sort_by_row(row_indices, column_indices, values);
}

// sort matrix elements by row index
template <typename Array1, typename Array2, typename Array3, typename IndexType, typename ValueType, typename MemorySpace>
void
coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
::sort_by_row_and_column(void)
{
    cusp::sort_by_row_and_column(row_indices, column_indices, values);
}

// determine whether matrix elements are sorted by row index
template <typename Array1, typename Array2, typename Array3, typename IndexType, typename ValueType, typename MemorySpace>
bool
coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
::is_sorted_by_row(void)
{
    return thrust::is_sorted(row_indices.begin(), row_indices.end());
}

// determine whether matrix elements are sorted by row and column index
template <typename Array1, typename Array2, typename Array3, typename IndexType, typename ValueType, typename MemorySpace>
bool
coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
::is_sorted_by_row_and_column(void)
{
    return thrust::is_sorted
           (thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), column_indices.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(row_indices.end(),   column_indices.end())));
}

} // end namespace cusp

