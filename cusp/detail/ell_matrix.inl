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

namespace cusp
{

//////////////////
// Constructors //
//////////////////
        
// construct empty matrix
template<typename IndexType, class SpaceOrAlloc>
ell_pattern<IndexType,SpaceOrAlloc>
    ::ell_pattern()
        : num_entries(0) {}

template <typename IndexType, typename ValueType, class SpaceOrAlloc>
ell_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::ell_matrix()
        : ell_pattern<IndexType,SpaceOrAlloc>() {}

// construct matrix with given shape and number of entries
template<typename IndexType, class SpaceOrAlloc>
ell_pattern<IndexType,SpaceOrAlloc>
    ::ell_pattern(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                  IndexType num_entries_per_row, IndexType stride)
        : column_indices(num_entries_per_row * stride),
          num_entries(num_entries), num_entries_per_row(num_entries_per_row), stride(stride),
          matrix_shape<IndexType>(num_rows, num_cols) {}

template <typename IndexType, typename ValueType, class SpaceOrAlloc>
ell_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::ell_matrix(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                 IndexType num_entries_per_row, IndexType stride)
        : values(num_entries_per_row * stride),
          ell_pattern<IndexType,SpaceOrAlloc>(num_rows, num_cols, num_entries, num_entries_per_row, stride) {}

// construct from another matrix
template<typename IndexType, class SpaceOrAlloc>
template <typename IndexType2, typename SpaceOrAlloc2>
ell_pattern<IndexType,SpaceOrAlloc>
    :: ell_pattern(const ell_pattern<IndexType2,SpaceOrAlloc2>& pattern)
        : column_indices(pattern.column_indices),
          num_entries(pattern.num_entries), num_entries_per_row(pattern.num_entries_per_row), stride(pattern.stride),
          matrix_shape<IndexType>(pattern) {}

template <typename IndexType, typename ValueType, class SpaceOrAlloc>
template <typename IndexType2, typename ValueType2, typename SpaceOrAlloc2>
ell_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::ell_matrix(const ell_matrix<IndexType2, ValueType2, SpaceOrAlloc2>& matrix)
        : values(matrix.values),
          ell_pattern<IndexType,SpaceOrAlloc>(matrix) {}

//////////////////////
// Member Functions //
//////////////////////

// resize matrix shape and storage
template <typename IndexType, class SpaceOrAlloc>
    void
    ell_pattern<IndexType,SpaceOrAlloc>
    ::resize(IndexType num_rows, IndexType num_cols, IndexType num_entries,
             IndexType num_entries_per_row, IndexType stride)
    {
        column_indices.resize(num_entries_per_row * stride);

        this->num_rows            = num_rows;
        this->num_cols            = num_cols;
        this->num_entries         = num_entries;
        this->num_entries_per_row = num_entries_per_row;
        this->stride              = stride;
    }

template <typename IndexType, typename ValueType, class SpaceOrAlloc>
    void
    ell_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::resize(IndexType num_rows, IndexType num_cols, IndexType num_entries,
             IndexType num_entries_per_row, IndexType stride)
    {
        values.resize(num_entries_per_row * stride);
        ell_pattern<IndexType,SpaceOrAlloc>::resize(num_rows, num_cols, num_entries,
                                                    num_entries_per_row, stride);
    }

// swap matrix contents
template <typename IndexType, class SpaceOrAlloc>
    void
    ell_pattern<IndexType,SpaceOrAlloc>
    ::swap(ell_pattern& pattern)
    {
        column_indices.swap(pattern.column_indices);

        thrust::swap(num_entries,         pattern.num_entries);
        thrust::swap(num_entries_per_row, pattern.num_entries_per_row);
        thrust::swap(stride,              pattern.stride);

        matrix_shape<IndexType>::swap(pattern);
    }

template <typename IndexType, typename ValueType, class SpaceOrAlloc>
    void
    ell_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::swap(ell_matrix& matrix)
    {
        ell_pattern<IndexType,SpaceOrAlloc>::swap(matrix);
        values.swap(matrix.values);
    }

} // end namespace cusp

