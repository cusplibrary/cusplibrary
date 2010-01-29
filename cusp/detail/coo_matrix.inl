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

#include <cusp/detail/convert.h>

namespace cusp
{

//////////////////
// Constructors //
//////////////////
        
// construct empty matrix
template<typename IndexType, class SpaceOrAlloc>
coo_pattern<IndexType,SpaceOrAlloc>
    ::coo_pattern() {}

template <typename IndexType, typename ValueType, class SpaceOrAlloc>
coo_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::coo_matrix()
        : coo_pattern<IndexType,SpaceOrAlloc>() {}

// construct matrix with given shape and number of entries
template<typename IndexType, class SpaceOrAlloc>
coo_pattern<IndexType,SpaceOrAlloc>
    :: coo_pattern(IndexType num_rows, IndexType num_cols, IndexType num_entries)
        : detail::matrix_base<IndexType>(num_rows, num_cols, num_entries),
          row_indices(num_entries), column_indices(num_entries) {}

template <typename IndexType, typename ValueType, class SpaceOrAlloc>
coo_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::coo_matrix(IndexType num_rows, IndexType num_cols, IndexType num_entries)
        : coo_pattern<IndexType,SpaceOrAlloc>(num_rows, num_cols, num_entries),
          values(num_entries) {}

// construct from another coo_matrix
template<typename IndexType, class SpaceOrAlloc>
template <typename IndexType2, typename SpaceOrAlloc2>
coo_pattern<IndexType,SpaceOrAlloc>
    :: coo_pattern(const coo_pattern<IndexType2,SpaceOrAlloc2>& pattern)
        : detail::matrix_base<IndexType>(pattern),
          row_indices(pattern.row_indices), column_indices(pattern.column_indices) {}

template <typename IndexType, typename ValueType, class SpaceOrAlloc>
template <typename IndexType2, typename ValueType2, typename SpaceOrAlloc2>
coo_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::coo_matrix(const coo_matrix<IndexType2, ValueType2, SpaceOrAlloc2>& matrix)
        : coo_pattern<IndexType,SpaceOrAlloc>(matrix),
          values(matrix.values) {}
        
// construct from a different matrix format
template <typename IndexType, typename ValueType, class SpaceOrAlloc>
template <typename MatrixType>
coo_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::coo_matrix(const MatrixType& matrix)
    {
        cusp::detail::convert(*this, matrix);
    }

//////////////////////
// Member Functions //
//////////////////////
        
// resize matrix shape and storage
template <typename IndexType, class SpaceOrAlloc>
    void
    coo_pattern<IndexType,SpaceOrAlloc>
    ::resize(IndexType num_rows, IndexType num_cols, IndexType num_entries)
    {
        this->num_rows    = num_rows;
        this->num_cols    = num_cols;
        this->num_entries = num_entries;

        row_indices.resize(num_entries);
        column_indices.resize(num_entries);
    }

template <typename IndexType, typename ValueType, class SpaceOrAlloc>
    void
    coo_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::resize(IndexType num_rows, IndexType num_cols, IndexType num_entries)
    {
        coo_pattern<IndexType,SpaceOrAlloc>::resize(num_rows, num_cols, num_entries);
        values.resize(num_entries);
    }

// swap matrix contents
template <typename IndexType, class SpaceOrAlloc>
    void
    coo_pattern<IndexType,SpaceOrAlloc>
    ::swap(coo_pattern& pattern)
    {
        detail::matrix_base<IndexType>::swap(pattern);

        row_indices.swap(pattern.row_indices);
        column_indices.swap(pattern.column_indices);
    }

template <typename IndexType, typename ValueType, class SpaceOrAlloc>
    void
    coo_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::swap(coo_matrix& matrix)
    {
        coo_pattern<IndexType,SpaceOrAlloc>::swap(matrix);
        values.swap(matrix.values);
    }


template <typename IndexType, typename ValueType, class SpaceOrAlloc>
template <typename IndexType2, typename ValueType2, typename SpaceOrAlloc2>
    coo_matrix<IndexType,ValueType,SpaceOrAlloc>&
    coo_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::operator=(const coo_matrix<IndexType2, ValueType2, SpaceOrAlloc2>& matrix)
    {
        // TODO use coo_pattern::operator= or coo_pattern::assign()
        this->num_rows       = matrix.num_rows;
        this->num_cols       = matrix.num_cols;
        this->num_entries    = matrix.num_entries;
        this->row_indices    = matrix.row_indices;
        this->column_indices = matrix.column_indices;
        this->values         = matrix.values;

        return *this;
    }


template <typename IndexType, typename ValueType, class SpaceOrAlloc>
template <typename MatrixType>
    coo_matrix<IndexType,ValueType,SpaceOrAlloc>&
    coo_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::operator=(const MatrixType& matrix)
    {
        cusp::detail::convert(*this, matrix);
        
        return *this;
    }

} // end namespace cusp

