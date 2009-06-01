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

#include <cusp/memory.h>
#include <cusp/matrix_shape.h>

namespace cusp
{
    struct row_major    {};
    struct column_major {};
    
    namespace detail
    {
        template <typename IndexType>
        IndexType index_of(const IndexType& i,        const IndexType& j, 
                           const IndexType& num_rows, const IndexType& num_cols,
                           row_major){
            return i * num_rows + j;
        }
            
        template <typename IndexType>
        IndexType index_of(const IndexType& i,        const IndexType& j, 
                           const IndexType& num_rows, const IndexType& num_cols,
                           column_major){
            return j * num_cols + i;
        }
    }


    // Definition //
    template<typename ValueType, class MemorySpace, class Orientation = cusp::row_major>
    struct dense_matrix : public matrix_shape<size_t>
    {
        typedef typename matrix_shape<size_t>::index_type index_type;
        typedef MemorySpace memory_space;
        typedef Orientation orientation;
        typedef ValueType * value_pointer;
        
        index_type num_entries;
        value_pointer values;
        
        
        template <typename IndexType>
        ValueType& operator()(const IndexType& i, const IndexType& j) { 
            return values[detail::index_of(i, j, (IndexType) num_rows, (IndexType) num_cols, orientation())];
        }

        template <typename IndexType>
        const ValueType& operator()(const IndexType& i, const IndexType& j) const { 
            return values[detail::index_of(i, j, (IndexType) num_rows, (IndexType) num_cols, orientation())];
        }
    };


    // Memory Management //
    template<typename ValueType, class MemorySpace, class Orientation>
    void allocate_matrix(dense_matrix<ValueType,MemorySpace,Orientation>& matrix,
                         size_t num_rows, size_t num_cols)
    {
        matrix.num_rows    = num_rows;
        matrix.num_cols    = num_cols;
        matrix.num_entries = num_rows * num_cols;

        matrix.values = cusp::new_array<ValueType,MemorySpace>(matrix.num_entries);
    }

    template<typename ValueType, class MemorySpace, class Orientation>
    void deallocate_matrix(dense_matrix<ValueType, MemorySpace, Orientation>& matrix)
    {
        cusp::delete_array<ValueType,MemorySpace>(matrix.values);
        matrix.values = 0;

        matrix.num_rows    = 0;
        matrix.num_cols    = 0;
        matrix.num_entries = 0;
    }

    template<typename ValueType, class MemorySpace1, class MemorySpace2, class Orientation>
    void memcpy_matrix(      dense_matrix<ValueType,MemorySpace1,Orientation>& dst,
                       const dense_matrix<ValueType,MemorySpace2,Orientation>& src)
    {
        dst.num_rows    = src.num_rows;
        dst.num_cols    = src.num_cols;
        dst.num_entries = src.num_entries;
        cusp::memcpy_array<ValueType,MemorySpace1,MemorySpace2>(dst.values, src.values, dst.num_entries);
    }

} // end namespace cusp

