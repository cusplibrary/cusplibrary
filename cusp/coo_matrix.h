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
    class coo_pattern : public matrix_shape<IndexType>
    {
        public:
        typedef typename matrix_shape<IndexType>::index_type index_type;

        typedef typename cusp::standard_memory_allocator<IndexType, SpaceOrAlloc>::type index_allocator_type;
        typedef typename cusp::allocator_space<index_allocator_type>::type memory_space;
        typedef typename cusp::coo_pattern<IndexType, SpaceOrAlloc> pattern_type;
        
        template<typename SpaceOrAlloc2>
        struct rebind { typedef coo_pattern<IndexType, SpaceOrAlloc2> type; };

        index_type num_entries;

        cusp::array1d<IndexType, index_allocator_type> row_indices;
        cusp::array1d<IndexType, index_allocator_type> column_indices;

        coo_pattern()
            : matrix_shape<IndexType>() {}

        coo_pattern(IndexType num_rows, IndexType num_cols, IndexType num_entries)
            : row_indices(num_entries), column_indices(num_entries),
              num_entries(num_entries),
              matrix_shape<IndexType>(num_rows, num_cols) {}

        template <typename IndexType2, typename SpaceOrAlloc2>
        coo_pattern(const coo_pattern<IndexType2,SpaceOrAlloc2>& pattern)
            : row_indices(pattern.row_indices), column_indices(pattern.column_indices),
              num_entries(pattern.num_entries),
              matrix_shape<IndexType>(pattern) {}
        
        void resize(IndexType num_rows, IndexType num_cols, IndexType num_entries)
        {
            row_indices.resize(num_entries);
            column_indices.resize(num_entries);

            this->num_rows    = num_rows;
            this->num_cols    = num_cols;
            this->num_entries = num_entries;
        }
        
        void swap(coo_pattern& pattern)
        {
            row_indices.swap(pattern.row_indices);
            column_indices.swap(pattern.column_indices);

            thrust::swap(num_entries, pattern.num_entries);
            
            matrix_shape<IndexType>::swap(pattern);
        }
    }; // class coo_pattern

    template <typename IndexType, typename ValueType, class SpaceOrAlloc>
    struct coo_matrix : public coo_pattern<IndexType, SpaceOrAlloc>
    {
        public:
        typedef typename cusp::standard_memory_allocator<ValueType, SpaceOrAlloc>::type value_allocator_type;
        typedef typename cusp::coo_matrix<IndexType, ValueType, SpaceOrAlloc> matrix_type;
    
        typedef ValueType value_type;
        
        template<typename SpaceOrAlloc2>
        struct rebind { typedef coo_matrix<IndexType, ValueType, SpaceOrAlloc2> type; };
    
        cusp::array1d<ValueType, value_allocator_type> values;
    
        // construct empty matrix
        coo_matrix()
            : coo_pattern<IndexType,SpaceOrAlloc>() {}
    
        // construct matrix with given shape and number of entries
        coo_matrix(IndexType num_rows, IndexType num_cols, IndexType num_entries)
            : values(num_entries),
              coo_pattern<IndexType,SpaceOrAlloc>(num_rows, num_cols, num_entries) {}
    
        // construct from another coo_matrix
        template <typename IndexType2, typename ValueType2, typename SpaceOrAlloc2>
        coo_matrix(const coo_matrix<IndexType2, ValueType2, SpaceOrAlloc2>& matrix)
            : values(matrix.values),
              coo_pattern<IndexType,SpaceOrAlloc>(matrix) {}

        void resize(IndexType num_rows, IndexType num_cols, IndexType num_entries)
        {
            values.resize(num_entries);
            coo_pattern<IndexType,SpaceOrAlloc>::resize(num_rows, num_cols, num_entries);
        }

        void swap(coo_matrix& matrix)
        {
            coo_pattern<IndexType,SpaceOrAlloc>::swap(matrix);
            values.swap(matrix.values);
        }

    }; // class coo_matrix

} // end namespace cusp

