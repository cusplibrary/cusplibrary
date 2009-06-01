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

#include <limits>

#include <cusp/memory.h>
#include <cusp/matrix_shape.h>

namespace cusp
{
    // Definition //
    template<typename IndexType, class MemorySpace>
    struct ell_pattern : public matrix_shape<IndexType>
    {
        typedef typename matrix_shape<IndexType>::index_type index_type;
        typedef index_type * index_pointer;
        typedef MemorySpace memory_space;
       
        const static index_type invalid_index = static_cast<IndexType>(-1);

        index_type num_entries;
        index_type num_entries_per_row;
        index_type stride;

        index_pointer column_indices;
    };


    template <typename IndexType, typename ValueType, class MemorySpace>
    struct ell_matrix : public ell_pattern<IndexType,MemorySpace>
    {
        typedef ell_pattern<IndexType,MemorySpace> pattern_type;

        typedef ValueType value_type;
        typedef value_type * value_pointer;

        value_pointer values;
    };
   

    // Memory Management //
    template<typename IndexType, class MemorySpace>
    void allocate_pattern(ell_pattern<IndexType, MemorySpace>& pattern,
                          const IndexType num_rows, const IndexType num_cols, const IndexType num_entries,
                          const IndexType num_entries_per_row, const IndexType stride)
    {
        pattern.num_rows            = num_rows;
        pattern.num_cols            = num_cols;
        pattern.num_entries         = num_entries;;
        pattern.num_entries_per_row = num_entries_per_row;
        pattern.stride              = stride;

        pattern.column_indices = cusp::new_array<IndexType,MemorySpace>(stride * num_entries_per_row);
    }
    
    template<typename IndexType, class MemorySpace1, class MemorySpace2>
    void allocate_pattern_like(      ell_pattern<IndexType,MemorySpace1>& pattern,
                               const ell_pattern<IndexType,MemorySpace2>& example)
    {
        allocate_pattern(pattern, example.num_rows, example.num_cols, example.num_entries, 
                                  example.num_entries_per_row, example.stride);
    }

    template<typename IndexType, typename ValueType, class MemorySpace>
    void allocate_matrix(ell_matrix<IndexType,ValueType,MemorySpace>& matrix,
                         const IndexType num_rows, const IndexType num_cols, const IndexType num_entries,
                         const IndexType num_entries_per_row, const IndexType stride)
    {
        allocate_pattern<IndexType,MemorySpace>(matrix, num_rows, num_cols, num_entries, num_entries_per_row, stride);
        matrix.values = cusp::new_array<ValueType,MemorySpace>(stride * num_entries_per_row);
    }

    template<typename IndexType, typename ValueType, class MemorySpace1, class MemorySpace2>
    void allocate_matrix_like(       ell_matrix<IndexType,ValueType,MemorySpace1>& matrix,
                               const ell_matrix<IndexType,ValueType,MemorySpace2>& example)
    {
        allocate_matrix(matrix, example.num_rows, example.num_cols, example.num_entries, 
                                example.num_entries_per_row, example.stride);
    }
    template<typename IndexType, class MemorySpace>
    void deallocate_pattern(ell_pattern<IndexType, MemorySpace>& pattern)
    {
        cusp::delete_array<IndexType,MemorySpace>(pattern.column_indices);

        pattern.num_rows            = 0;
        pattern.num_cols            = 0;
        pattern.num_entries         = 0;
        pattern.num_entries_per_row = 0;
        pattern.stride              = 0;

        pattern.column_indices = 0;
    }

    template<typename IndexType, typename ValueType, class MemorySpace>
    void deallocate_matrix(ell_matrix<IndexType,ValueType,MemorySpace>& matrix)
    {
        cusp::delete_array<ValueType,MemorySpace>(matrix.values);
        matrix.values = 0;
        deallocate_pattern(matrix);
    }

    template<typename IndexType, class MemorySpace1, class MemorySpace2>
    void memcpy_pattern(      ell_pattern<IndexType, MemorySpace1>& dst,
                        const ell_pattern<IndexType, MemorySpace2>& src)
    {
        dst.num_rows            = src.num_rows;
        dst.num_cols            = src.num_cols;
        dst.num_entries         = src.num_entries;
        dst.num_entries_per_row = src.num_entries_per_row;
        dst.stride              = src.stride;

        cusp::memcpy_array<IndexType,MemorySpace1,MemorySpace2>(dst.column_indices, src.column_indices, dst.num_entries_per_row * dst.stride);
    }

    template<typename IndexType, typename ValueType, class MemorySpace1, class MemorySpace2>
    void memcpy_matrix(      ell_matrix<IndexType,ValueType,MemorySpace1>& dst,
                       const ell_matrix<IndexType,ValueType,MemorySpace2>& src)
    {
        cusp::memcpy_pattern<IndexType,MemorySpace1,MemorySpace2>(dst, src);
        cusp::memcpy_array<ValueType,MemorySpace1,MemorySpace2>(dst.values, src.values, dst.num_entries_per_row * dst.stride);
    }

} // end namespace cusp
