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
    struct row_major    {};
    struct column_major {};
    
    namespace detail
    {
        template <typename IndexType>
        IndexType minor_dimension(IndexType num_rows, IndexType num_cols, row_major)    { return num_cols; }
        
        template <typename IndexType>
        IndexType minor_dimension(IndexType num_rows, IndexType num_cols, column_major) { return num_rows; }

        template <typename IndexType>
        IndexType major_dimension(IndexType num_rows, IndexType num_cols, row_major)    { return num_rows; }
        
        template <typename IndexType>
        IndexType major_dimension(IndexType num_rows, IndexType num_cols, column_major) { return num_cols; }

        template <typename IndexType>
        IndexType index_of(IndexType i, IndexType j, IndexType num_rows, IndexType num_cols, row_major)    { return i * num_cols + j; }
            
        template <typename IndexType>
        IndexType index_of(IndexType i, IndexType j, IndexType num_rows, IndexType num_cols, column_major) { return j * num_rows + i; }
    }

    template<typename ValueType, class SpaceOrAlloc, class Orientation = cusp::row_major>
    struct array2d : public detail::matrix_base<size_t>
    {
        public:
        typedef size_t    index_type;
        typedef ValueType value_type;
        
        typedef typename cusp::choose_memory_allocator<ValueType, SpaceOrAlloc>::type value_allocator_type;
        typedef typename cusp::allocator_space<value_allocator_type>::type memory_space;

        typedef Orientation orientation;
        
        template<typename SpaceOrAlloc2>
        struct rebind { typedef array2d<ValueType, SpaceOrAlloc2, Orientation> type; };
       
        cusp::array1d<ValueType, value_allocator_type> values;
       
        // construct empty matrix
        array2d();

        // construct matrix with given shape and number of entries
        array2d(size_t num_rows, size_t num_cols);
        
        // construct from another array2d (with the same Orientation)
        template <typename ValueType2, typename SpaceOrAlloc2>
        array2d(const array2d<ValueType2, SpaceOrAlloc2, Orientation>& matrix);
        
        // construct from a different matrix format
        template <typename MatrixType>
        array2d(const MatrixType& matrix);
        
        typename value_allocator_type::reference operator()(const index_type i, const index_type j)
        { 
            return values[detail::index_of(i, j, num_rows, num_cols, orientation())];
        }

        typename value_allocator_type::const_reference operator()(const index_type i, const index_type j) const
        { 
            return values[detail::index_of(i, j, num_rows, num_cols, orientation())];
        }
        
        void resize(index_type num_rows, index_type num_cols);

        void swap(array2d& matrix);
        
        template <typename ValueType2, typename SpaceOrAlloc2>
        array2d& operator=(const array2d<ValueType2, SpaceOrAlloc2, Orientation>& matrix);

        template <typename MatrixType>
        array2d& operator=(const MatrixType& matrix);

    }; // class array2d

} // end namespace cusp

#include <cusp/detail/array2d.inl>

