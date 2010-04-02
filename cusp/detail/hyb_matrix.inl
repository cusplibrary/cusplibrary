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


#include <cusp/ell_matrix.h>
#include <cusp/coo_matrix.h>

namespace cusp
{

//////////////////
// Constructors //
//////////////////
        
// construct empty matrix
template <typename IndexType, typename ValueType, class MemorySpace>
hyb_matrix<IndexType,ValueType,MemorySpace>
    ::hyb_matrix() {}

// construct matrix with given shape and number of entries
template <typename IndexType, typename ValueType, class MemorySpace>
hyb_matrix<IndexType,ValueType,MemorySpace>
    ::hyb_matrix(IndexType num_rows, IndexType num_cols,
                 IndexType num_ell_entries, IndexType num_coo_entries,
                 IndexType num_entries_per_row, IndexType alignment)
        : detail::matrix_base<IndexType,ValueType,MemorySpace>(num_rows, num_cols, num_ell_entries + num_coo_entries),
          ell(num_rows, num_cols, num_ell_entries, num_entries_per_row, alignment),
          coo(num_rows, num_cols, num_coo_entries) {}

// construct from another matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename IndexType2, typename ValueType2, typename MemorySpace2>
hyb_matrix<IndexType,ValueType,MemorySpace>
    ::hyb_matrix(const hyb_matrix<IndexType2, ValueType2, MemorySpace2>& matrix)
        : detail::matrix_base<IndexType,ValueType,MemorySpace>(matrix.num_rows, matrix.num_cols, matrix.num_entries),
          ell(matrix.ell),
          coo(matrix.coo) {}

// construct from a different matrix format
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
hyb_matrix<IndexType,ValueType,MemorySpace>
    ::hyb_matrix(const MatrixType& matrix)
    {
        cusp::detail::convert(*this, matrix);
    }

//////////////////////
// Member Functions //
//////////////////////
        
// resize matrix shape and storage
template <typename IndexType, typename ValueType, class MemorySpace>
    void
    hyb_matrix<IndexType,ValueType,MemorySpace>
    ::resize(IndexType num_rows, IndexType num_cols,
             IndexType num_ell_entries, IndexType num_coo_entries,
             IndexType num_entries_per_row, IndexType alignment)
    {
            this->num_rows    = num_rows;
            this->num_cols    = num_cols;
            this->num_entries = num_ell_entries + num_coo_entries;

            ell.resize(num_rows, num_cols, num_ell_entries, num_entries_per_row, alignment);
            coo.resize(num_rows, num_cols, num_coo_entries);
    }

// swap matrix contents
template <typename IndexType, typename ValueType, class MemorySpace>
    void
    hyb_matrix<IndexType,ValueType,MemorySpace>
    ::swap(hyb_matrix& matrix)
    {
        detail::matrix_base<IndexType,ValueType,MemorySpace>::swap(matrix);

        ell.swap(matrix.ell);
        coo.swap(matrix.coo);
    }

template <typename IndexType, typename ValueType, class MemorySpace>
template <typename IndexType2, typename ValueType2, typename MemorySpace2>
    hyb_matrix<IndexType,ValueType,MemorySpace>&
    hyb_matrix<IndexType,ValueType,MemorySpace>
    ::operator=(const hyb_matrix<IndexType2, ValueType2, MemorySpace2>& matrix)
    {
        this->num_rows    = matrix.num_rows;
        this->num_cols    = matrix.num_cols;
        this->num_entries = matrix.num_entries;
        this->ell         = matrix.ell;
        this->coo         = matrix.coo;

        return *this;
    }

template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
    hyb_matrix<IndexType,ValueType,MemorySpace>&
    hyb_matrix<IndexType,ValueType,MemorySpace>
    ::operator=(const MatrixType& matrix)
    {
        cusp::detail::convert(*this, matrix);
        
        return *this;
    }

} // end namespace cusp

