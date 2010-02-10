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
template<typename IndexType, class MemorySpace>
csr_pattern<IndexType,MemorySpace>
    ::csr_pattern() {}

template <typename IndexType, typename ValueType, class MemorySpace>
csr_matrix<IndexType,ValueType,MemorySpace>
    ::csr_matrix()
        : csr_pattern<IndexType,MemorySpace>() {}

// construct matrix with given shape and number of entries
template<typename IndexType, class MemorySpace>
csr_pattern<IndexType,MemorySpace>
    :: csr_pattern(IndexType num_rows, IndexType num_cols, IndexType num_entries)
        : detail::matrix_base<IndexType>(num_rows, num_cols, num_entries),
          row_offsets(num_rows + 1), column_indices(num_entries) {}

template <typename IndexType, typename ValueType, class MemorySpace>
csr_matrix<IndexType,ValueType,MemorySpace>
    ::csr_matrix(IndexType num_rows, IndexType num_cols, IndexType num_entries)
        : csr_pattern<IndexType,MemorySpace>(num_rows, num_cols, num_entries),
          values(num_entries) {}

// construct from another matrix
template<typename IndexType, class MemorySpace>
template <typename IndexType2, typename MemorySpace2>
csr_pattern<IndexType,MemorySpace>
    :: csr_pattern(const csr_pattern<IndexType2,MemorySpace2>& pattern)
        : detail::matrix_base<IndexType>(pattern),
          row_offsets(pattern.row_offsets), column_indices(pattern.column_indices) {}

template <typename IndexType, typename ValueType, class MemorySpace>
template <typename IndexType2, typename ValueType2, typename MemorySpace2>
csr_matrix<IndexType,ValueType,MemorySpace>
    ::csr_matrix(const csr_matrix<IndexType2, ValueType2, MemorySpace2>& matrix)
        : csr_pattern<IndexType,MemorySpace>(matrix),
          values(matrix.values) {}

// construct from a different matrix format
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
csr_matrix<IndexType,ValueType,MemorySpace>
    ::csr_matrix(const MatrixType& matrix)
    {
        cusp::detail::convert(*this, matrix);
    }

//////////////////////
// Member Functions //
//////////////////////

// resize matrix shape and storage
template <typename IndexType, class MemorySpace>
    void
    csr_pattern<IndexType,MemorySpace>
    ::resize(IndexType num_rows, IndexType num_cols, IndexType num_entries)
    {
        this->num_rows    = num_rows;
        this->num_cols    = num_cols;
        this->num_entries = num_entries;

        row_offsets.resize(num_rows + 1);
        column_indices.resize(num_entries);
    }

template <typename IndexType, typename ValueType, class MemorySpace>
    void
    csr_matrix<IndexType,ValueType,MemorySpace>
    ::resize(IndexType num_rows, IndexType num_cols, IndexType num_entries)
    {
        csr_pattern<IndexType,MemorySpace>::resize(num_rows, num_cols, num_entries);
        values.resize(num_entries);
    }

// swap matrix contents
template <typename IndexType, class MemorySpace>
    void
    csr_pattern<IndexType,MemorySpace>
    ::swap(csr_pattern& pattern)
    {
        detail::matrix_base<IndexType>::swap(pattern);
        row_offsets.swap(pattern.row_offsets);
        column_indices.swap(pattern.column_indices);
    }

template <typename IndexType, typename ValueType, class MemorySpace>
    void
    csr_matrix<IndexType,ValueType,MemorySpace>
    ::swap(csr_matrix& matrix)
    {
        csr_pattern<IndexType,MemorySpace>::swap(matrix);
        values.swap(matrix.values);
    }

template <typename IndexType, typename ValueType, class MemorySpace>
template <typename IndexType2, typename ValueType2, typename MemorySpace2>
    csr_matrix<IndexType,ValueType,MemorySpace>&
    csr_matrix<IndexType,ValueType,MemorySpace>
    ::operator=(const csr_matrix<IndexType2, ValueType2, MemorySpace2>& matrix)
    {
        // TODO use csr_pattern::operator= or csr_pattern::assign()
        this->num_rows       = matrix.num_rows;
        this->num_cols       = matrix.num_cols;
        this->num_entries    = matrix.num_entries;
        this->row_offsets    = matrix.row_offsets;
        this->column_indices = matrix.column_indices;
        this->values         = matrix.values;

        return *this;
    }


template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
    csr_matrix<IndexType,ValueType,MemorySpace>&
    csr_matrix<IndexType,ValueType,MemorySpace>
    ::operator=(const MatrixType& matrix)
    {
        cusp::detail::convert(*this, matrix);
        
        return *this;
    }
} // end namespace cusp

