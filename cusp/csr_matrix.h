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

    template<typename IndexType, class SpaceOrAlloc>
    class csr_pattern : public detail::matrix_base<IndexType>
    {
        public:
        typedef IndexType index_type;

        typedef typename cusp::choose_memory_allocator<IndexType, SpaceOrAlloc>::type index_allocator_type;
        typedef typename cusp::allocator_space<index_allocator_type>::type memory_space;
        typedef typename cusp::csr_pattern<IndexType, SpaceOrAlloc> pattern_type;
       
        template<typename SpaceOrAlloc2>
        struct rebind { typedef csr_pattern<IndexType, SpaceOrAlloc2> type; };

        cusp::array1d<IndexType, index_allocator_type> row_offsets;
        cusp::array1d<IndexType, index_allocator_type> column_indices;
    
        csr_pattern();
    
        csr_pattern(IndexType num_rows, IndexType num_cols, IndexType num_entries);
        
        template <typename IndexType2, typename SpaceOrAlloc2>
        csr_pattern(const csr_pattern<IndexType2,SpaceOrAlloc2>& pattern);

        void resize(IndexType num_rows, IndexType num_cols, IndexType num_entries);
        
        void swap(csr_pattern& pattern);
    }; // class csr_pattern

    template <typename IndexType, typename ValueType, class SpaceOrAlloc>
    class csr_matrix : public csr_pattern<IndexType, SpaceOrAlloc>
    {
        public:
        typedef typename cusp::choose_memory_allocator<ValueType, SpaceOrAlloc>::type value_allocator_type;
        typedef typename cusp::csr_matrix<IndexType, ValueType, SpaceOrAlloc> matrix_type;
    
        typedef ValueType value_type;
        
        template<typename SpaceOrAlloc2>
        struct rebind { typedef csr_matrix<IndexType, ValueType, SpaceOrAlloc2> type; };
    
        cusp::array1d<ValueType, value_allocator_type> values;
    
        // construct empty matrix
        csr_matrix();
    
        // construct matrix with given shape and number of entries
        csr_matrix(IndexType num_rows, IndexType num_cols, IndexType num_entries);
    
        // construct from another csr_matrix
        template <typename IndexType2, typename ValueType2, typename SpaceOrAlloc2>
        csr_matrix(const csr_matrix<IndexType2, ValueType2, SpaceOrAlloc2>& matrix);
        
        // construct from a different matrix format
        template <typename MatrixType>
        csr_matrix(const MatrixType& matrix);
        
        // sparse matrix-vector multiplication
        template <typename VectorType1, typename VectorType2>
        void multiply(const VectorType1& x, VectorType2& y) const;
        
        void resize(IndexType num_rows, IndexType num_cols, IndexType num_entries);

        void swap(csr_matrix& matrix);
        
        template <typename IndexType2, typename ValueType2, typename SpaceOrAlloc2>
        csr_matrix& operator=(const csr_matrix<IndexType2, ValueType2, SpaceOrAlloc2>& matrix);

        template <typename MatrixType>
        csr_matrix& operator=(const MatrixType& matrix);
    }; // class csr_matrix
            
} // end namespace cusp

#include <cusp/detail/csr_matrix.inl>

