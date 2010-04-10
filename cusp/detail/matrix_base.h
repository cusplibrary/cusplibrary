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

#include <thrust/utility.h>

namespace cusp
{
namespace detail
{

    template<typename IndexType, typename ValueType, typename MemorySpace>
    class matrix_base
    {
        public:
            typedef IndexType   index_type;
            typedef ValueType   value_type;
            typedef MemorySpace memory_space;

            index_type num_rows;
            index_type num_cols;
            index_type num_entries;
            
            matrix_base()
                : num_rows(0), num_cols(0), num_entries(0) {}

            matrix_base(IndexType rows, IndexType cols, IndexType entries)
                : num_rows(rows), num_cols(cols), num_entries(entries) {}

            void swap(matrix_base& base)
            {
                thrust::swap(num_rows,    base.num_rows);
                thrust::swap(num_cols,    base.num_cols);
                thrust::swap(num_entries, base.num_entries);
            }
    };

} // end namespace detail
} // end namespace cusp

