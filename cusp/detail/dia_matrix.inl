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
#include <cusp/detail/utils.h>

#include <cusp/detail/spmv.h>

namespace cusp
{

//////////////////
// Constructors //
//////////////////
        
// construct empty matrix
template <typename IndexType, typename ValueType, class SpaceOrAlloc>
dia_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::dia_matrix() {}

// construct matrix with given shape and number of entries
template <typename IndexType, typename ValueType, class SpaceOrAlloc>
dia_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::dia_matrix(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                 IndexType num_diagonals, IndexType alignment)
        : detail::matrix_base<IndexType>(num_rows, num_cols, num_entries),
          diagonal_offsets(num_diagonals),
          values(detail::round_up(num_rows, alignment), num_diagonals) {}

// construct from another dia_matrix
template <typename IndexType, typename ValueType, class SpaceOrAlloc>
template <typename IndexType2, typename ValueType2, typename SpaceOrAlloc2>
dia_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::dia_matrix(const dia_matrix<IndexType2, ValueType2, SpaceOrAlloc2>& matrix)
        : detail::matrix_base<IndexType>(matrix),
          diagonal_offsets(matrix.diagonal_offsets), values(matrix.values) {}

// construct from a different matrix format
template <typename IndexType, typename ValueType, class SpaceOrAlloc>
template <typename MatrixType>
dia_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::dia_matrix(const MatrixType& matrix)
    {
        cusp::detail::convert(*this, matrix);
    }

//////////////////////
// Member Functions //
//////////////////////

// sparse matrix-vector multiplication
template <typename IndexType, typename ValueType, class SpaceOrAlloc>
template <typename VectorType1, typename VectorType2>
    void
    dia_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::multiply(const VectorType1& x, VectorType2& y) const
    {
        cusp::detail::spmv(*this, x, y);
    }

// resize matrix shape and storage
template <typename IndexType, typename ValueType, class SpaceOrAlloc>
    void
    dia_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::resize(IndexType num_rows, IndexType num_cols, IndexType num_entries,
             IndexType num_diagonals, IndexType alignment)
    {
        this->num_rows      = num_rows;
        this->num_cols      = num_cols;
        this->num_entries   = num_entries;

        diagonal_offsets.resize(num_diagonals);
        values.resize(detail::round_up(num_rows, alignment), num_diagonals);
    }

// swap matrix contents
template <typename IndexType, typename ValueType, class SpaceOrAlloc>
    void
    dia_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::swap(dia_matrix& matrix)
    {
        detail::matrix_base<IndexType>::swap(matrix);

        diagonal_offsets.swap(matrix.diagonal_offsets);
        values.swap(matrix.values);
    }

// copy another dia_matrix
template <typename IndexType, typename ValueType, class SpaceOrAlloc>
template <typename IndexType2, typename ValueType2, typename SpaceOrAlloc2>
    dia_matrix<IndexType,ValueType,SpaceOrAlloc>&
    dia_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::operator=(const dia_matrix<IndexType2, ValueType2, SpaceOrAlloc2>& matrix)
    {
        this->num_rows         = matrix.num_rows;
        this->num_cols         = matrix.num_cols;
        this->num_entries      = matrix.num_entries;
        this->diagonal_offsets = matrix.diagonal_offsets;
        this->values           = matrix.values;

        return *this;
    }

// copy a matrix in a different format
template <typename IndexType, typename ValueType, class SpaceOrAlloc>
template <typename MatrixType>
    dia_matrix<IndexType,ValueType,SpaceOrAlloc>&
    dia_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::operator=(const MatrixType& matrix)
    {
        cusp::detail::convert(*this, matrix);
        
        return *this;
    }

} // end namespace cusp

