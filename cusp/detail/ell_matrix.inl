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
template<typename IndexType, class SpaceOrAlloc>
ell_pattern<IndexType,SpaceOrAlloc>
    ::ell_pattern() {}

template <typename IndexType, typename ValueType, class SpaceOrAlloc>
ell_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::ell_matrix()
        : ell_pattern<IndexType,SpaceOrAlloc>() {}

// construct matrix with given shape and number of entries
template<typename IndexType, class SpaceOrAlloc>
ell_pattern<IndexType,SpaceOrAlloc>
    ::ell_pattern(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                  IndexType num_entries_per_row, IndexType alignment)
        : detail::matrix_base<IndexType>(num_rows, num_cols, num_entries),
          column_indices(detail::round_up(num_rows, alignment), num_entries_per_row) {}

template <typename IndexType, typename ValueType, class SpaceOrAlloc>
ell_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::ell_matrix(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                 IndexType num_entries_per_row, IndexType alignment)
        : ell_pattern<IndexType,SpaceOrAlloc>(num_rows, num_cols, num_entries, num_entries_per_row, alignment),
          values(detail::round_up(num_rows, alignment), num_entries_per_row) {}

// construct from another matrix
template<typename IndexType, class SpaceOrAlloc>
template <typename IndexType2, typename SpaceOrAlloc2>
ell_pattern<IndexType,SpaceOrAlloc>
    :: ell_pattern(const ell_pattern<IndexType2,SpaceOrAlloc2>& pattern)
        : detail::matrix_base<IndexType>(pattern),
          column_indices(pattern.column_indices) {}

template <typename IndexType, typename ValueType, class SpaceOrAlloc>
template <typename IndexType2, typename ValueType2, typename SpaceOrAlloc2>
ell_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::ell_matrix(const ell_matrix<IndexType2, ValueType2, SpaceOrAlloc2>& matrix)
        : ell_pattern<IndexType,SpaceOrAlloc>(matrix),
          values(matrix.values) {}

// construct from a different matrix format
template <typename IndexType, typename ValueType, class SpaceOrAlloc>
template <typename MatrixType>
ell_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::ell_matrix(const MatrixType& matrix)
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
    ell_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::multiply(const VectorType1& x, VectorType2& y) const
    {
        cusp::detail::spmv(*this, x, y);
    }

// resize matrix shape and storage
template <typename IndexType, class SpaceOrAlloc>
    void
    ell_pattern<IndexType,SpaceOrAlloc>
    ::resize(IndexType num_rows, IndexType num_cols, IndexType num_entries,
             IndexType num_entries_per_row, IndexType alignment)
    {
        this->num_rows    = num_rows;
        this->num_cols    = num_cols;
        this->num_entries = num_entries;

        column_indices.resize(detail::round_up(num_rows, alignment), num_entries_per_row);
    }

template <typename IndexType, typename ValueType, class SpaceOrAlloc>
    void
    ell_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::resize(IndexType num_rows, IndexType num_cols, IndexType num_entries,
             IndexType num_entries_per_row, IndexType alignment)
    {
        ell_pattern<IndexType,SpaceOrAlloc>::resize(num_rows, num_cols, num_entries,
                                                    num_entries_per_row, alignment);

        values.resize(detail::round_up(num_rows, alignment), num_entries_per_row);
    }

// swap matrix contents
template <typename IndexType, class SpaceOrAlloc>
    void
    ell_pattern<IndexType,SpaceOrAlloc>
    ::swap(ell_pattern& pattern)
    {
        detail::matrix_base<IndexType>::swap(pattern);

        column_indices.swap(pattern.column_indices);
    }

template <typename IndexType, typename ValueType, class SpaceOrAlloc>
    void
    ell_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::swap(ell_matrix& matrix)
    {
        ell_pattern<IndexType,SpaceOrAlloc>::swap(matrix);
        values.swap(matrix.values);
    }

template <typename IndexType, typename ValueType, class SpaceOrAlloc>
template <typename IndexType2, typename ValueType2, typename SpaceOrAlloc2>
    ell_matrix<IndexType,ValueType,SpaceOrAlloc>&
    ell_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::operator=(const ell_matrix<IndexType2, ValueType2, SpaceOrAlloc2>& matrix)
    {
        // TODO use ell_pattern::operator= or ell_pattern::assign()
        this->num_rows            = matrix.num_rows;
        this->num_cols            = matrix.num_cols;
        this->num_entries         = matrix.num_entries;
        this->column_indices      = matrix.column_indices;
        this->values              = matrix.values;

        return *this;
    }

template <typename IndexType, typename ValueType, class SpaceOrAlloc>
template <typename MatrixType>
    ell_matrix<IndexType,ValueType,SpaceOrAlloc>&
    ell_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::operator=(const MatrixType& matrix)
    {
        cusp::detail::convert(*this, matrix);
        
        return *this;
    }

} // end namespace cusp

