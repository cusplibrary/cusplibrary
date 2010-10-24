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
#include <cusp/detail/utils.h>

namespace cusp
{

//////////////////
// Constructors //
//////////////////
        
// construct empty matrix
template <typename IndexType, typename ValueType, class MemorySpace>
dia_matrix<IndexType,ValueType,MemorySpace>
    ::dia_matrix() {}

// construct matrix with given shape and number of entries
template <typename IndexType, typename ValueType, class MemorySpace>
dia_matrix<IndexType,ValueType,MemorySpace>
    ::dia_matrix(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                 IndexType num_diagonals, IndexType alignment)
        : detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::dia_format>(num_rows, num_cols, num_entries),
          diagonal_offsets(num_diagonals)
    {
      // TODO use array2d constructor when it can accept pitch
      values.resize(num_rows, num_diagonals, detail::round_up(num_rows, alignment));
    }

// construct from a different matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
dia_matrix<IndexType,ValueType,MemorySpace>
    ::dia_matrix(const MatrixType& matrix)
    {
        cusp::convert(matrix, *this);
    }

//////////////////////
// Member Functions //
//////////////////////

// resize matrix shape and storage
template <typename IndexType, typename ValueType, class MemorySpace>
    void
    dia_matrix<IndexType,ValueType,MemorySpace>
    ::resize(IndexType num_rows, IndexType num_cols, IndexType num_entries,
             IndexType num_diagonals, IndexType alignment)
    {
        this->num_rows      = num_rows;
        this->num_cols      = num_cols;
        this->num_entries   = num_entries;

        diagonal_offsets.resize(num_diagonals);
        values.resize(num_rows, num_diagonals, detail::round_up(num_rows, alignment));
    }

// swap matrix contents
template <typename IndexType, typename ValueType, class MemorySpace>
    void
    dia_matrix<IndexType,ValueType,MemorySpace>
    ::swap(dia_matrix& matrix)
    {
        detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::dia_format>::swap(matrix);

        diagonal_offsets.swap(matrix.diagonal_offsets);
        values.swap(matrix.values);
    }

// copy a matrix in a different format
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
    dia_matrix<IndexType,ValueType,MemorySpace>&
    dia_matrix<IndexType,ValueType,MemorySpace>
    ::operator=(const MatrixType& matrix)
    {
        cusp::convert(matrix, *this);
        
        return *this;
    }

} // end namespace cusp

