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

    template <typename IndexType, typename ValueType, class SpaceOrAlloc>
    class dia_matrix : public matrix_shape<IndexType>
    {
        public:

        // TODO statically assert is_signed<IndexType>
        typedef IndexType index_type;
        typedef ValueType value_type;

        typedef typename cusp::standard_memory_allocator<IndexType, SpaceOrAlloc>::type index_allocator_type;
        typedef typename cusp::standard_memory_allocator<ValueType, SpaceOrAlloc>::type value_allocator_type;
        typedef typename cusp::allocator_space<index_allocator_type>::type memory_space;
        
        template<typename SpaceOrAlloc2>
        struct rebind { typedef dia_matrix<IndexType, ValueType, SpaceOrAlloc2> type; };

        index_type num_entries;
        index_type num_diagonals;
        index_type stride;

        cusp::array1d<IndexType, index_allocator_type> diagonal_offsets;
        cusp::array1d<ValueType, value_allocator_type> values;
            
        dia_matrix()
            : matrix_shape<IndexType>(),
              num_entries(0), num_diagonals(0), stride(0) {}

        dia_matrix(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                   IndexType num_diagonals, IndexType stride)
            : matrix_shape<IndexType>(num_rows, num_cols),
              num_entries(num_entries), num_diagonals(num_diagonals), stride(stride),
              diagonal_offsets(num_diagonals), values(num_diagonals * stride) {}
        
        template <typename IndexType2, typename ValueType2, typename SpaceOrAlloc2>
        dia_matrix(const dia_matrix<IndexType2, ValueType2, SpaceOrAlloc2>& matrix)
            : matrix_shape<IndexType>(matrix),
              num_entries(matrix.num_entries), num_diagonals(matrix.num_diagonals), stride(matrix.stride),
              diagonal_offsets(matrix.diagonal_offsets), values(matrix.values) {}
        
        void resize(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                    IndexType num_diagonals, IndexType stride)
        {
            diagonal_offsets.resize(num_diagonals);
            values.resize(num_diagonals * stride);

            this->num_rows      = num_rows;
            this->num_cols      = num_cols;
            this->num_entries   = num_entries;
            this->num_diagonals = num_diagonals;
            this->stride        = stride;
        }

        void swap(dia_matrix& matrix)
        {
            diagonal_offsets.swap(matrix.diagonal_offsets);
            values.swap(matrix.values);

            thrust::swap(num_entries,   matrix.num_entries);
            thrust::swap(num_diagonals, matrix.num_diagonals);
            thrust::swap(stride,        matrix.stride);
            
            matrix_shape<IndexType>::swap(matrix);
        }
    }; // class dia_matrix
    
} // end namespace cusp

