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
#include <cusp/ell_matrix.h>
#include <cusp/coo_matrix.h>

namespace cusp
{
    // Definition //
    template <typename IndexType, typename ValueType, class MemorySpace>
    struct hyb_matrix : public matrix_shape<IndexType>
    {
        typedef IndexType   index_type;
        typedef ValueType   value_type;
        typedef MemorySpace memory_space;
   
        index_type num_entries;

        cusp::ell_matrix<IndexType,ValueType,MemorySpace> ell;
        cusp::coo_matrix<IndexType,ValueType,MemorySpace> coo;
    };
   

    // Memory Management //
    template<typename IndexType, typename ValueType, class MemorySpace>
    void allocate_matrix(hyb_matrix<IndexType,ValueType,MemorySpace>& matrix,
                         const IndexType num_rows, const IndexType num_cols,
                         const IndexType num_ell_entries, const IndexType num_coo_entries,
                         const IndexType num_entries_per_row, const IndexType stride)
    {
        matrix.num_rows    = num_rows;
        matrix.num_cols    = num_cols;
        matrix.num_entries = num_ell_entries + num_coo_entries;
        
        cusp::allocate_matrix(matrix.ell, num_rows, num_cols, num_ell_entries, num_entries_per_row, stride);
        cusp::allocate_matrix(matrix.coo, num_rows, num_cols, num_coo_entries);
    }


    template<typename IndexType, typename ValueType, class MemorySpace1, class MemorySpace2>
    void allocate_matrix_like(      hyb_matrix<IndexType,ValueType,MemorySpace1>& matrix,
                              const hyb_matrix<IndexType,ValueType,MemorySpace2>& example)
    {
        cusp::allocate_matrix(matrix, example.num_rows, example.num_cols,
                                      example.ell.num_entries, example.coo.num_entries,
                                      example.ell.num_entries_per_row, example.ell.stride);
    }


    template<typename IndexType, typename ValueType, class MemorySpace>
    void deallocate_matrix(hyb_matrix<IndexType,ValueType,MemorySpace>& matrix)
    {
        cusp::deallocate_matrix(matrix.ell);
        cusp::deallocate_matrix(matrix.coo);
        
        matrix.num_rows    = 0;
        matrix.num_cols    = 0;
        matrix.num_entries = 0;
    }

    template<typename IndexType, typename ValueType, class MemorySpace1, class MemorySpace2>
    void memcpy_matrix(      hyb_matrix<IndexType,ValueType,MemorySpace1>& dst,
                       const hyb_matrix<IndexType,ValueType,MemorySpace2>& src)
    {
        dst.num_rows    = src.num_rows;
        dst.num_cols    = src.num_cols;
        dst.num_entries = src.num_entries;

        cusp::memcpy_matrix(dst.ell, src.ell);
        cusp::memcpy_matrix(dst.coo, src.coo);
    }

} // end namespace cusp
