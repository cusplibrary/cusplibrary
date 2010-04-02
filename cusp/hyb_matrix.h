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
/*! \addtogroup container_classes Container Classes 
 *  \addtogroup sparse_matrix_formats Sparse Matrices
 *  \ingroup container_classes
 *  \{
 */
    // Forward definitions
    template <typename IndexType, typename ValueType, class MemorySpace> class ell_matrix;
    template <typename IndexType, typename ValueType, class MemorySpace> class coo_matrix;

    // TODO document
    template <typename IndexType, typename ValueType, class MemorySpace>
    class hyb_matrix : public detail::matrix_base<IndexType,ValueType,MemorySpace>
    {
        public:
        template<typename MemorySpace2>
        struct rebind { typedef hyb_matrix<IndexType, ValueType, MemorySpace2> type; };

        cusp::ell_matrix<IndexType,ValueType,MemorySpace> ell;
        cusp::coo_matrix<IndexType,ValueType,MemorySpace> coo;

        // construct empty matrix
        hyb_matrix();

        // construct matrix with given shape and number of entries
        hyb_matrix(IndexType num_rows, IndexType num_cols,
                   IndexType num_ell_entries, IndexType num_coo_entries,
                   IndexType num_entries_per_row, IndexType alignment = 16);

        // construct from another hyb_matrix
        template <typename IndexType2, typename ValueType2, typename MemorySpace2>
        hyb_matrix(const hyb_matrix<IndexType2, ValueType2, MemorySpace2>& matrix);
        
        // construct from a different matrix format
        template <typename MatrixType>
        hyb_matrix(const MatrixType& matrix);
        
        void resize(IndexType num_rows, IndexType num_cols,
                    IndexType num_ell_entries, IndexType num_coo_entries,
                    IndexType num_entries_per_row, IndexType alignment = 16);

        void swap(hyb_matrix& matrix);
        
        template <typename IndexType2, typename ValueType2, typename MemorySpace2>
        hyb_matrix& operator=(const hyb_matrix<IndexType2, ValueType2, MemorySpace2>& matrix);

        template <typename MatrixType>
        hyb_matrix& operator=(const MatrixType& matrix);
    }; // class hyb_matrix
/*! \}
 */

} // end namespace cusp

#include <cusp/detail/hyb_matrix.inl>

