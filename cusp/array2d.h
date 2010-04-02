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
        __host__ __device__
        IndexType minor_dimension(IndexType num_rows, IndexType num_cols, row_major)    { return num_cols; }
        
        template <typename IndexType>
        __host__ __device__
        IndexType minor_dimension(IndexType num_rows, IndexType num_cols, column_major) { return num_rows; }

        template <typename IndexType>
        __host__ __device__
        IndexType major_dimension(IndexType num_rows, IndexType num_cols, row_major)    { return num_rows; }
        
        template <typename IndexType>
        __host__ __device__
        IndexType major_dimension(IndexType num_rows, IndexType num_cols, column_major) { return num_cols; }

        template <typename IndexType>
        __host__ __device__
        IndexType linear_index_to_row_index(IndexType linear_index, IndexType num_rows, IndexType num_cols, row_major)    { return linear_index / num_cols; }
            
        template <typename IndexType>
        __host__ __device__
        IndexType linear_index_to_col_index(IndexType linear_index, IndexType num_rows, IndexType num_cols, row_major)    { return linear_index % num_cols; }
        
        template <typename IndexType>
        __host__ __device__
        IndexType linear_index_to_row_index(IndexType linear_index, IndexType num_rows, IndexType num_cols, column_major)    { return linear_index % num_rows; }
            
        template <typename IndexType>
        __host__ __device__
        IndexType linear_index_to_col_index(IndexType linear_index, IndexType num_rows, IndexType num_cols, column_major)    { return linear_index / num_rows; }

        template <typename IndexType>
        __host__ __device__
        IndexType index_of(IndexType i, IndexType j, IndexType num_rows, IndexType num_cols, row_major)    { return i * num_cols + j; }
            
        template <typename IndexType>
        __host__ __device__
        IndexType index_of(IndexType i, IndexType j, IndexType num_rows, IndexType num_cols, column_major) { return j * num_rows + i; }
    }

    template<typename ValueType, class MemorySpace, class Orientation = cusp::row_major>
    struct array2d : public detail::matrix_base<int,ValueType,MemorySpace>
    {
        public:
        typedef typename cusp::choose_memory_allocator<ValueType, MemorySpace>::type value_allocator_type;

        typedef Orientation orientation;
        
        template<typename MemorySpace2>
        struct rebind { typedef array2d<ValueType, MemorySpace2, Orientation> type; };
       
        cusp::array1d<ValueType, value_allocator_type> values;
       
        // construct empty matrix
        array2d();

        // construct matrix with given shape and number of entries
        array2d(int num_rows, int num_cols);
        
        // construct matrix with given shape and number of entries and fill with a given value
        array2d(int num_rows, int num_cols, const ValueType& value);
        
        // construct from another array2d (with the same Orientation)
        template <typename ValueType2, typename MemorySpace2>
        array2d(const array2d<ValueType2, MemorySpace2, Orientation>& matrix);
        
        // construct from a different matrix format
        template <typename MatrixType>
        array2d(const MatrixType& matrix);
        
        typename value_allocator_type::reference operator()(const int i, const int j)
        { 
            return values[detail::index_of(i, j, this->num_rows, this->num_cols, orientation())];
        }

        typename value_allocator_type::const_reference operator()(const int i, const int j) const
        { 
            return values[detail::index_of(i, j, this->num_rows, this->num_cols, orientation())];
        }
        
        void resize(int num_rows, int num_cols);

        void swap(array2d& matrix);
        
        template <typename ValueType2, typename MemorySpace2>
        array2d& operator=(const array2d<ValueType2, MemorySpace2, Orientation>& matrix);

        template <typename MatrixType>
        array2d& operator=(const MatrixType& matrix);

    }; // class array2d

} // end namespace cusp

#include <cusp/detail/array2d.inl>

