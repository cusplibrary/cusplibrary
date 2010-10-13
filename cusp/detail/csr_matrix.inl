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
template <typename IndexType, typename ValueType, class MemorySpace>
csr_matrix<IndexType,ValueType,MemorySpace>
    ::csr_matrix()
        : detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::csr_format>() {}

// construct matrix with given shape and number of entries
template <typename IndexType, typename ValueType, class MemorySpace>
csr_matrix<IndexType,ValueType,MemorySpace>
    ::csr_matrix(IndexType num_rows, IndexType num_cols, IndexType num_entries)
        : detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::csr_format>(num_rows, num_cols, num_entries),
          row_offsets(num_rows + 1), column_indices(num_entries), values(num_entries) {}

// construct from a different matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
csr_matrix<IndexType,ValueType,MemorySpace>
    ::csr_matrix(const MatrixType& matrix)
    {
        cusp::detail::convert(matrix, *this);
    }

//////////////////////
// Member Functions //
//////////////////////

// resize matrix shape and storage
template <typename IndexType, typename ValueType, class MemorySpace>
    void
    csr_matrix<IndexType,ValueType,MemorySpace>
    ::resize(IndexType num_rows, IndexType num_cols, IndexType num_entries)
    {
        this->num_rows    = num_rows;
        this->num_cols    = num_cols;
        this->num_entries = num_entries;

        row_offsets.resize(num_rows + 1);
        column_indices.resize(num_entries);
        values.resize(num_entries);
    }

// swap matrix contents
template <typename IndexType, typename ValueType, class MemorySpace>
    void
    csr_matrix<IndexType,ValueType,MemorySpace>
    ::swap(csr_matrix& matrix)
    {
        detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::csr_format>::swap(matrix);
        row_offsets.swap(matrix.row_offsets);
        column_indices.swap(matrix.column_indices);
        values.swap(matrix.values);
    }

// assignment from another matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
    csr_matrix<IndexType,ValueType,MemorySpace>&
    csr_matrix<IndexType,ValueType,MemorySpace>
    ::operator=(const MatrixType& matrix)
    {
        cusp::detail::convert(matrix, *this);
        
        return *this;
    }
} // end namespace cusp

