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

#include <cusp/ell_matrix.h>
#include <cusp/coo_matrix.h>

namespace cusp
{

    template <typename IndexType, typename ValueType, class SpaceOrAlloc>
    class hyb_matrix : public matrix_shape<IndexType>
    {
        public:
        typedef IndexType   index_type;
        typedef ValueType   value_type;

        typedef typename cusp::standard_memory_allocator<IndexType, SpaceOrAlloc>::type index_allocator_type;
        typedef typename cusp::standard_memory_allocator<ValueType, SpaceOrAlloc>::type value_allocator_type;
        typedef typename cusp::allocator_space<index_allocator_type>::type memory_space;

        template<typename SpaceOrAlloc2>
        struct rebind { typedef hyb_matrix<IndexType, ValueType, SpaceOrAlloc2> type; };

        index_type num_entries;

        cusp::ell_matrix<IndexType,ValueType,SpaceOrAlloc> ell;
        cusp::coo_matrix<IndexType,ValueType,SpaceOrAlloc> coo;

        hyb_matrix();

        hyb_matrix(IndexType num_rows, IndexType num_cols,
                   IndexType num_ell_entries, IndexType num_coo_entries,
                   IndexType num_entries_per_row, IndexType stride);

        // construct from another coo_matrix
        template <typename IndexType2, typename ValueType2, typename SpaceOrAlloc2>
        hyb_matrix(const hyb_matrix<IndexType2, ValueType2, SpaceOrAlloc2>& matrix);
        
        void resize(IndexType num_rows, IndexType num_cols,
                    IndexType num_ell_entries, IndexType num_coo_entries,
                    IndexType num_entries_per_row, IndexType stride);

        void swap(hyb_matrix& matrix);
    }; // class hyb_matrix

} // end namespace cusp

#include <cusp/detail/hyb_matrix.inl>

