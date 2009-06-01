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
    // Definition //
    template<typename IndexType, class MemorySpace>
    struct coo_pattern : public matrix_shape<IndexType>
    {
        typedef typename matrix_shape<IndexType>::index_type index_type;
        typedef index_type * index_pointer;
        typedef MemorySpace memory_space;
        
        index_type num_entries;

        index_pointer row_indices;
        index_pointer column_indices;
    };


    template <typename IndexType, typename ValueType, class MemorySpace>
    struct coo_matrix : public coo_pattern<IndexType,MemorySpace>
    {
        typedef coo_pattern<IndexType,MemorySpace> pattern_type;

        typedef ValueType value_type;
        typedef value_type * value_pointer;

        value_pointer values;
    };
   

    // Memory Management //
    template<typename IndexType, class MemorySpace>
    void allocate_pattern(coo_pattern<IndexType, MemorySpace>& pattern,
                          IndexType num_rows, IndexType num_cols, IndexType num_entries)
    {
        pattern.num_rows    = num_rows;
        pattern.num_cols    = num_cols;
        pattern.num_entries = num_entries;

        pattern.row_indices    = cusp::new_array<IndexType,MemorySpace>(num_entries);
        pattern.column_indices = cusp::new_array<IndexType,MemorySpace>(num_entries);
    }

    template<typename IndexType, class MemorySpace1, class MemorySpace2>
    void allocate_pattern_like(      coo_pattern<IndexType,MemorySpace1>& pattern,
                               const coo_pattern<IndexType,MemorySpace2>& example)
    {
        allocate_pattern(pattern, example.num_rows, example.num_cols, example.num_entries);
    }

    template<typename IndexType, typename ValueType, class MemorySpace>
    void allocate_matrix(coo_matrix<IndexType,ValueType,MemorySpace>& matrix,
                         IndexType num_rows, IndexType num_cols, IndexType num_entries)
    {
        allocate_pattern<IndexType,MemorySpace>(matrix, num_rows, num_cols, num_entries);
        matrix.values = cusp::new_array<ValueType,MemorySpace>(num_entries);
    }
    
    template<typename IndexType, typename ValueType, class MemorySpace1, class MemorySpace2>
    void allocate_matrix_like(      coo_matrix<IndexType,ValueType,MemorySpace1>& matrix,
                              const coo_matrix<IndexType,ValueType,MemorySpace2>& example)
    {
        allocate_matrix(matrix, example.num_rows, example.num_cols, example.num_entries);
    }

    template<typename IndexType, class MemorySpace>
    void deallocate_pattern(coo_pattern<IndexType, MemorySpace>& pattern)
    {
        cusp::delete_array<IndexType,MemorySpace>(pattern.row_indices);
        cusp::delete_array<IndexType,MemorySpace>(pattern.column_indices);

        pattern.num_rows       = 0;
        pattern.num_cols       = 0;
        pattern.num_entries    = 0;
        pattern.row_indices    = 0;
        pattern.column_indices = 0;
    }

    template<typename IndexType, typename ValueType, class MemorySpace>
    void deallocate_matrix(coo_matrix<IndexType,ValueType,MemorySpace>& matrix)
    {
        cusp::delete_array<ValueType,MemorySpace>(matrix.values);
        matrix.values = 0;
        deallocate_pattern(matrix);
    }

    template<typename IndexType, class MemorySpace1, class MemorySpace2>
    void memcpy_pattern(      coo_pattern<IndexType, MemorySpace1>& dst,
                        const coo_pattern<IndexType, MemorySpace2>& src)
    {
        dst.num_rows    = src.num_rows;
        dst.num_cols    = src.num_cols;
        dst.num_entries = src.num_entries;
        cusp::memcpy_array<IndexType,MemorySpace1,MemorySpace2>(dst.row_indices,    src.row_indices,    dst.num_entries);
        cusp::memcpy_array<IndexType,MemorySpace1,MemorySpace2>(dst.column_indices, src.column_indices, dst.num_entries);
    }

    template<typename IndexType, typename ValueType, class MemorySpace1, class MemorySpace2>
    void memcpy_matrix(      coo_matrix<IndexType,ValueType,MemorySpace1>& dst,
                       const coo_matrix<IndexType,ValueType,MemorySpace2>& src)
    {
        cusp::memcpy_pattern<IndexType,MemorySpace1,MemorySpace2>(dst, src);
        cusp::memcpy_array<ValueType,MemorySpace1,MemorySpace2>(dst.values, src.values, dst.num_entries);
    }

} // end namespace cusp
