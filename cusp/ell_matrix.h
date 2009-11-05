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

#include <cusp/detail/matrix_base.h>

namespace cusp
{
    // Forward definitions
    struct column_major;
    template<typename ValueType, class SpaceOrAlloc, class Orientation> class array2d;

    template<typename IndexType, class SpaceOrAlloc>
    class ell_pattern : public detail::matrix_base<IndexType>
    {
        public:
        typedef IndexType index_type;

        typedef typename cusp::choose_memory_allocator<IndexType, SpaceOrAlloc>::type index_allocator_type;
        typedef typename cusp::allocator_space<index_allocator_type>::type memory_space;
        typedef typename cusp::ell_pattern<IndexType, SpaceOrAlloc> pattern_type;

        const static index_type invalid_index = static_cast<IndexType>(-1);
        
        template<typename SpaceOrAlloc2>
        struct rebind { typedef ell_pattern<IndexType, SpaceOrAlloc2> type; };

        cusp::array2d<IndexType, index_allocator_type, cusp::column_major> column_indices;

        ell_pattern();
                   
        ell_pattern(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                    IndexType num_entries_per_row, IndexType alignment = 16);

        template <typename IndexType2, typename SpaceOrAlloc2>
        ell_pattern(const ell_pattern<IndexType2,SpaceOrAlloc2>& pattern);

        void resize(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                    IndexType num_entries_per_row, IndexType alignment = 16);

        void swap(ell_pattern& pattern);
    }; // class ell_pattern

    template <typename IndexType, typename ValueType, class SpaceOrAlloc>
    class ell_matrix : public ell_pattern<IndexType, SpaceOrAlloc>
    {
        public:
        typedef typename cusp::choose_memory_allocator<ValueType, SpaceOrAlloc>::type value_allocator_type;
        typedef typename cusp::ell_matrix<IndexType, ValueType, SpaceOrAlloc> matrix_type;
    
        typedef ValueType value_type;
    
        template<typename SpaceOrAlloc2>
        struct rebind { typedef ell_matrix<IndexType, ValueType, SpaceOrAlloc2> type; };

        cusp::array2d<ValueType, value_allocator_type, cusp::column_major> values;
    
        // construct empty matrix
        ell_matrix();
    
        // construct matrix with given shape and number of entries
        ell_matrix(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                   IndexType num_entries_per_row, IndexType alignment = 16);
    
        // construct from another ell_matrix
        template <typename IndexType2, typename ValueType2, typename SpaceOrAlloc2>
        ell_matrix(const ell_matrix<IndexType2, ValueType2, SpaceOrAlloc2>& matrix);
        
        // construct from a different matrix format
        template <typename MatrixType>
        ell_matrix(const MatrixType& matrix);
        
        void resize(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                    IndexType num_entries_per_row, IndexType alignment = 16);

        void swap(ell_matrix& matrix);
        
        template <typename IndexType2, typename ValueType2, typename SpaceOrAlloc2>
        ell_matrix& operator=(const ell_matrix<IndexType2, ValueType2, SpaceOrAlloc2>& matrix);

        template <typename MatrixType>
        ell_matrix& operator=(const MatrixType& matrix);
    }; // class ell_matrix

} // end namespace cusp

#include <cusp/array2d.h>

#include <cusp/detail/ell_matrix.inl>

