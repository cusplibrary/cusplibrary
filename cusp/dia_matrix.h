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

        typedef typename cusp::choose_memory_allocator<IndexType, SpaceOrAlloc>::type index_allocator_type;
        typedef typename cusp::choose_memory_allocator<ValueType, SpaceOrAlloc>::type value_allocator_type;
        typedef typename cusp::allocator_space<index_allocator_type>::type memory_space;
        
        template<typename SpaceOrAlloc2>
        struct rebind { typedef dia_matrix<IndexType, ValueType, SpaceOrAlloc2> type; };

        index_type num_entries;
        index_type num_diagonals;
        index_type stride;

        cusp::array1d<IndexType, index_allocator_type> diagonal_offsets;
        cusp::array1d<ValueType, value_allocator_type> values;
            
        // construct empty matrix
        dia_matrix();

        // construct matrix with given shape and number of entries
        dia_matrix(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                   IndexType num_diagonals, IndexType stride);
        
        // construct from another dia_matrix
        template <typename IndexType2, typename ValueType2, typename SpaceOrAlloc2>
        dia_matrix(const dia_matrix<IndexType2, ValueType2, SpaceOrAlloc2>& matrix);
        
        // construct from a different matrix format
        template <typename MatrixType>
        dia_matrix(const MatrixType& matrix);

        void resize(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                    IndexType num_diagonals, IndexType stride);

        void swap(dia_matrix& matrix);
        
        template <typename IndexType2, typename ValueType2, typename SpaceOrAlloc2>
        dia_matrix& operator=(const dia_matrix<IndexType2, ValueType2, SpaceOrAlloc2>& matrix);

        template <typename MatrixType>
        dia_matrix& operator=(const MatrixType& matrix);
    }; // class dia_matrix
    
} // end namespace cusp

#include <cusp/detail/dia_matrix.inl>

