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


#pragma once

#include <cusp/array1d.h>
#include <cusp/matrix_shape.h>

namespace cusp
{

    template<typename IndexType, class SpaceOrAlloc>
    class ell_pattern : public matrix_shape<IndexType>
    {
        public:
        typedef typename matrix_shape<IndexType>::index_type index_type;

        typedef typename cusp::standard_memory_allocator<IndexType, SpaceOrAlloc>::type index_allocator_type;
        typedef typename cusp::allocator_space<index_allocator_type>::type memory_space;
        typedef typename cusp::ell_pattern<IndexType, SpaceOrAlloc> pattern_type;

        const static index_type invalid_index = static_cast<IndexType>(-1);
        
        template<typename SpaceOrAlloc2>
        struct rebind { typedef ell_pattern<IndexType, SpaceOrAlloc2> type; };

        index_type num_entries;
        index_type num_entries_per_row;
        index_type stride;

        cusp::array1d<IndexType, index_allocator_type> column_indices;

        ell_pattern()
            : matrix_shape<IndexType>(0,0) {}
                   
        ell_pattern(IndexType num_rows, IndexType num_cols, IndexType num_entries, IndexType num_entries_per_row, IndexType stride)
            : column_indices(num_entries_per_row * stride),
              num_entries(num_entries), num_entries_per_row(num_entries_per_row), stride(stride),
              matrix_shape<IndexType>(num_rows, num_cols) {}

        template <typename IndexType2, typename SpaceOrAlloc2>
        ell_pattern(const ell_pattern<IndexType2,SpaceOrAlloc2>& pattern)
            : column_indices(pattern.column_indices),
              num_entries(pattern.num_entries), num_entries_per_row(pattern.num_entries_per_row), stride(pattern.stride),
              matrix_shape<IndexType>(pattern) {}

        void resize(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                    IndexType num_entries_per_row, IndexType stride)
        {
            column_indices.resize(num_entries_per_row * stride);

            this->num_rows            = num_rows;
            this->num_cols            = num_cols;
            this->num_entries         = num_entries;
            this->num_entries_per_row = num_entries_per_row;
            this->stride              = stride;
        }

        void swap(ell_pattern& pattern)
        {
            column_indices.swap(pattern.column_indices);

            thrust::swap(num_entries,         pattern.num_entries);
            thrust::swap(num_entries_per_row, pattern.num_entries_per_row);
            thrust::swap(stride,              pattern.stride);
            
            matrix_shape<IndexType>::swap(pattern);
        }
    }; // class ell_pattern

    template <typename IndexType, typename ValueType, class SpaceOrAlloc>
    struct ell_matrix : public ell_pattern<IndexType, SpaceOrAlloc>
    {
        public:
        typedef typename cusp::standard_memory_allocator<ValueType, SpaceOrAlloc>::type value_allocator_type;
        typedef typename cusp::ell_matrix<IndexType, ValueType, SpaceOrAlloc> matrix_type;
    
        typedef ValueType value_type;
    
        template<typename SpaceOrAlloc2>
        struct rebind { typedef ell_matrix<IndexType, ValueType, SpaceOrAlloc2> type; };

        cusp::array1d<ValueType, value_allocator_type> values;
    
        // construct empty matrix
        ell_matrix()
            : ell_pattern<IndexType,SpaceOrAlloc>() {}
    
        // construct matrix with given shape and number of entries
        ell_matrix(IndexType num_rows, IndexType num_cols, IndexType num_entries, IndexType num_entries_per_row, IndexType stride)
            : values(num_entries_per_row * stride),
              ell_pattern<IndexType,SpaceOrAlloc>(num_rows, num_cols, num_entries, num_entries_per_row, stride) {}
    
        // construct from another ell_matrix
        template <typename IndexType2, typename ValueType2, typename SpaceOrAlloc2>
        ell_matrix(const ell_matrix<IndexType2, ValueType2, SpaceOrAlloc2>& matrix)
            : values(matrix.values),
              ell_pattern<IndexType,SpaceOrAlloc>(matrix) {}
        
        void resize(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                    IndexType num_entries_per_row, IndexType stride)
        {
            values.resize(num_entries_per_row * stride);
            ell_pattern<IndexType,SpaceOrAlloc>::resize(num_rows, num_cols, num_entries,
                                                        num_entries_per_row, stride);
        }

        void swap(ell_matrix& matrix)
        {
            ell_pattern<IndexType,SpaceOrAlloc>::swap(matrix);
            values.swap(matrix.values);
        }
    }; // class ell_matrix

} // end namespace cusp

