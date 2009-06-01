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
    template <typename IndexType, typename ValueType, class MemorySpace>
    struct dia_matrix : public matrix_shape<IndexType>
    {
        typedef typename matrix_shape<IndexType>::index_type index_type;
        typedef ValueType value_type;
        typedef int offset_type;  // diagonal_offsets must be signed 
        typedef MemorySpace memory_space;
        typedef value_type  * value_pointer;
        typedef offset_type * offset_pointer;
        
        index_type num_entries;
        index_type num_diagonals;
        index_type stride;

        offset_pointer diagonal_offsets;
        value_pointer  values;
    };
   

    // Memory Management //
    template<typename IndexType, typename ValueType, class MemorySpace>
    void allocate_matrix(dia_matrix<IndexType,ValueType,MemorySpace>& matrix,
                         IndexType num_rows, IndexType num_cols, IndexType num_entries,
                         IndexType num_diagonals, IndexType stride)
    {
        typedef typename dia_matrix<IndexType,ValueType,MemorySpace>::offset_type offset_type;

        matrix.num_rows = num_rows;
        matrix.num_cols = num_cols;
        matrix.num_entries = num_entries;
        matrix.num_diagonals = num_diagonals;
        matrix.stride = stride;

        matrix.diagonal_offsets = cusp::new_array<offset_type,MemorySpace>(num_diagonals);
        matrix.values           = cusp::new_array<ValueType,MemorySpace>(num_diagonals * stride);
    }
    
    template<typename IndexType, typename ValueType, class MemorySpace1, class MemorySpace2>
    void allocate_matrix_like(      dia_matrix<IndexType,ValueType,MemorySpace1>& matrix,
                              const dia_matrix<IndexType,ValueType,MemorySpace2>& example)
    {
        allocate_matrix(matrix, example.num_rows, example.num_cols, example.num_entries, 
                                example.num_diagonals, example.stride);
    }

    template<typename IndexType, typename ValueType, class MemorySpace>
    void deallocate_matrix(dia_matrix<IndexType,ValueType,MemorySpace>& matrix)
    {
        typedef typename dia_matrix<IndexType,ValueType,MemorySpace>::offset_type offset_type;

        cusp::delete_array<offset_type,MemorySpace>(matrix.diagonal_offsets);
        cusp::delete_array<ValueType,MemorySpace>(matrix.values);

        matrix.num_rows         = 0;
        matrix.num_cols         = 0;
        matrix.num_entries      = 0;
        matrix.num_diagonals    = 0;
        matrix.stride           = 0;
        matrix.diagonal_offsets = 0;
        matrix.values = 0;
    }

    template<typename IndexType, typename ValueType, class MemorySpace1, class MemorySpace2>
    void memcpy_matrix(      dia_matrix<IndexType,ValueType,MemorySpace1>& dst,
                       const dia_matrix<IndexType,ValueType,MemorySpace2>& src)
    {
        typedef typename dia_matrix<IndexType,ValueType,MemorySpace1>::offset_type offset_type;

        dst.num_rows      = src.num_rows;
        dst.num_cols      = src.num_cols;
        dst.num_entries   = src.num_entries;
        dst.num_diagonals = src.num_diagonals;
        dst.stride        = src.stride;
        cusp::memcpy_array<offset_type,MemorySpace1,MemorySpace2>(dst.diagonal_offsets, src.diagonal_offsets, dst.num_diagonals);
        cusp::memcpy_array<ValueType,MemorySpace1,MemorySpace2>(dst.values, src.values, dst.num_diagonals * dst.stride);
    }

} // end namespace cusp
