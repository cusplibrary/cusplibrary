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

#include <cusp/convert.h>

#include <cusp/detail/format_utils.h>
#include <thrust/iterator/zip_iterator.h>

#if (THRUST_VERSION < 100300)
#include <thrust/is_sorted.h>
#else
#include <thrust/sort.h>
#endif

namespace cusp
{

//////////////////
// Constructors //
//////////////////
        
// construct empty matrix
template <typename IndexType, typename ValueType, class MemorySpace>
coo_matrix<IndexType,ValueType,MemorySpace>
    ::coo_matrix()
        : detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::coo_format>() {}

// construct matrix with given shape and number of entries
template <typename IndexType, typename ValueType, class MemorySpace>
coo_matrix<IndexType,ValueType,MemorySpace>
    ::coo_matrix(IndexType num_rows, IndexType num_cols, IndexType num_entries)
        : detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::coo_format>(num_rows, num_cols, num_entries),
          row_indices(num_entries), column_indices(num_entries), values(num_entries) {}

// construct from a different matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
coo_matrix<IndexType,ValueType,MemorySpace>
    ::coo_matrix(const MatrixType& matrix)
    {
        cusp::convert(matrix, *this);
    }

//////////////////////
// Member Functions //
//////////////////////
        
// resize matrix shape and storage
template <typename IndexType, typename ValueType, class MemorySpace>
    void
    coo_matrix<IndexType,ValueType,MemorySpace>
    ::resize(IndexType num_rows, IndexType num_cols, IndexType num_entries)
    {
        this->num_rows    = num_rows;
        this->num_cols    = num_cols;
        this->num_entries = num_entries;

        row_indices.resize(num_entries);
        column_indices.resize(num_entries);
        values.resize(num_entries);
    }

// swap matrix contents
template <typename IndexType, typename ValueType, class MemorySpace>
    void
    coo_matrix<IndexType,ValueType,MemorySpace>
    ::swap(coo_matrix& matrix)
    {
        detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::coo_format>::swap(matrix);

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
        cusp::detail::sort_by_row(row_indices, column_indices, values);
    }

// sort matrix elements by row index
template <typename IndexType, typename ValueType, class MemorySpace>
    void
    coo_matrix<IndexType,ValueType,MemorySpace>
    ::sort_by_row_and_column(void)
    {
        cusp::detail::sort_by_row_and_column(row_indices, column_indices, values);
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

} // end namespace cusp

