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

namespace cusp
{

//////////////////
// Constructors //
//////////////////
        
// construct empty matrix
template <typename IndexType, typename ValueType, class SpaceOrAlloc>
dia_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::dia_matrix()
        : num_entries(0), num_diagonals(0), stride(0) {}

// construct matrix with given shape and number of entries
template <typename IndexType, typename ValueType, class SpaceOrAlloc>
dia_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::dia_matrix(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                 IndexType num_diagonals, IndexType stride)
        : matrix_shape<IndexType>(num_rows, num_cols),
          num_entries(num_entries), num_diagonals(num_diagonals), stride(stride),
          diagonal_offsets(num_diagonals), values(num_diagonals * stride) {}

// construct from another dia_matrix
template <typename IndexType, typename ValueType, class SpaceOrAlloc>
template <typename IndexType2, typename ValueType2, typename SpaceOrAlloc2>
dia_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::dia_matrix(const dia_matrix<IndexType2, ValueType2, SpaceOrAlloc2>& matrix)
        : matrix_shape<IndexType>(matrix),
          num_entries(matrix.num_entries), num_diagonals(matrix.num_diagonals), stride(matrix.stride),
          diagonal_offsets(matrix.diagonal_offsets), values(matrix.values) {}

// construct from a different matrix format
template <typename IndexType, typename ValueType, class SpaceOrAlloc>
template <typename MatrixType>
dia_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::dia_matrix(const MatrixType& matrix)
    {
        cusp::convert(*this, matrix);
    }

//////////////////////
// Member Functions //
//////////////////////

// resize matrix shape and storage
template <typename IndexType, typename ValueType, class SpaceOrAlloc>
    void
    dia_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::resize(IndexType num_rows, IndexType num_cols, IndexType num_entries,
             IndexType num_diagonals, IndexType stride)
    {
        diagonal_offsets.resize(num_diagonals);
        values.resize(num_diagonals * stride);

        this->num_entries   = num_entries;
        this->num_diagonals = num_diagonals;
        this->stride        = stride;
        this->num_rows      = num_rows;
        this->num_cols      = num_cols;
    }

// swap matrix contents
template <typename IndexType, typename ValueType, class SpaceOrAlloc>
    void
    dia_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::swap(dia_matrix& matrix)
    {
        diagonal_offsets.swap(matrix.diagonal_offsets);
        values.swap(matrix.values);

        thrust::swap(num_entries,   matrix.num_entries);
        thrust::swap(num_diagonals, matrix.num_diagonals);
        thrust::swap(stride,        matrix.stride);
        matrix_shape<IndexType>::swap(matrix);
    }

template <typename IndexType, typename ValueType, class SpaceOrAlloc>
template <typename IndexType2, typename ValueType2, typename SpaceOrAlloc2>
    dia_matrix<IndexType,ValueType,SpaceOrAlloc>&
    dia_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::operator=(const dia_matrix<IndexType2, ValueType2, SpaceOrAlloc2>& matrix)
    {
        // TODO use coo_pattern::operator= or coo_pattern::assign()
        this->values           = matrix.values;
        this->diagonal_offsets = matrix.diagonal_offsets;
        this->num_diagonals    = matrix.num_diagonals;
        this->stride           = matrix.stride;
        this->num_entries      = matrix.num_entries;
        this->num_rows         = matrix.num_rows;
        this->num_cols         = matrix.num_cols;

        return *this;
    }


template <typename IndexType, typename ValueType, class SpaceOrAlloc>
template <typename MatrixType>
    dia_matrix<IndexType,ValueType,SpaceOrAlloc>&
    dia_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::operator=(const MatrixType& matrix)
    {
        cusp::convert(*this, matrix);
        
        return *this;
    }

} // end namespace cusp

