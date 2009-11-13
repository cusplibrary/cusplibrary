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

#include <cusp/detail/config.h>

#include <cusp/array1d.h>
#include <cusp/detail/matrix_base.h>

namespace cusp
{

    // Forward definitions
    struct column_major;
    template<typename ValueType, class SpaceOrAlloc, class Orientation> class array2d;

    template <typename IndexType, typename ValueType, class SpaceOrAlloc>
    class dia_matrix : public detail::matrix_base<IndexType>
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

        cusp::array1d<IndexType, index_allocator_type>                     diagonal_offsets;
        cusp::array2d<ValueType, value_allocator_type, cusp::column_major> values;
            
        // construct empty matrix
        dia_matrix();

        // construct matrix with given shape and number of entries
        dia_matrix(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                   IndexType num_diagonals, IndexType alignment = 16);
        
        // construct from another dia_matrix
        template <typename IndexType2, typename ValueType2, typename SpaceOrAlloc2>
        dia_matrix(const dia_matrix<IndexType2, ValueType2, SpaceOrAlloc2>& matrix);
        
        // construct from a different matrix format
        template <typename MatrixType>
        dia_matrix(const MatrixType& matrix);
        
        // sparse matrix-vector multiplication
        template <typename VectorType1, typename VectorType2>
        void operator()(const VectorType1& x, VectorType2& y) const;

        void resize(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                    IndexType num_diagonals, IndexType alignment = 16);

        void swap(dia_matrix& matrix);
        
        template <typename IndexType2, typename ValueType2, typename SpaceOrAlloc2>
        dia_matrix& operator=(const dia_matrix<IndexType2, ValueType2, SpaceOrAlloc2>& matrix);

        template <typename MatrixType>
        dia_matrix& operator=(const MatrixType& matrix);
    }; // class dia_matrix
    
} // end namespace cusp

#include <cusp/array2d.h>

#include <cusp/detail/dia_matrix.inl>

