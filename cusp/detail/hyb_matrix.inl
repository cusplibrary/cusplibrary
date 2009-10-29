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

namespace cusp
{

//////////////////
// Constructors //
//////////////////
        
// construct empty matrix
template <typename IndexType, typename ValueType, class SpaceOrAlloc>
hyb_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::hyb_matrix()
        : num_entries(0),  
          ell(), coo() {}

// construct matrix with given shape and number of entries
template <typename IndexType, typename ValueType, class SpaceOrAlloc>
hyb_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::hyb_matrix(IndexType num_rows, IndexType num_cols,
                 IndexType num_ell_entries, IndexType num_coo_entries,
                 IndexType num_entries_per_row, IndexType stride)
        : matrix_shape<IndexType>(num_rows, num_cols),
          num_entries(num_ell_entries + num_coo_entries),
          ell(num_rows, num_cols, num_ell_entries, num_entries_per_row, stride),
          coo(num_rows, num_cols, num_coo_entries) {}

// construct from another matrix
template <typename IndexType, typename ValueType, class SpaceOrAlloc>
template <typename IndexType2, typename ValueType2, typename SpaceOrAlloc2>
hyb_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::hyb_matrix(const hyb_matrix<IndexType2, ValueType2, SpaceOrAlloc2>& matrix)
        : matrix_shape<IndexType>(matrix),
          num_entries(matrix.num_entries),
          ell(matrix.ell),
          coo(matrix.coo) {}

//////////////////////
// Member Functions //
//////////////////////

// resize matrix shape and storage
template <typename IndexType, typename ValueType, class SpaceOrAlloc>
    void
    hyb_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::resize(IndexType num_rows, IndexType num_cols,
             IndexType num_ell_entries, IndexType num_coo_entries,
             IndexType num_entries_per_row, IndexType stride)
    {
            ell.resize(num_rows, num_cols, num_ell_entries, num_entries_per_row, stride);
            coo.resize(num_rows, num_cols, num_coo_entries);

            this->num_rows    = num_rows;
            this->num_cols    = num_cols;
            this->num_entries = num_ell_entries + num_coo_entries;
    }

// swap matrix contents
template <typename IndexType, typename ValueType, class SpaceOrAlloc>
    void
    hyb_matrix<IndexType,ValueType,SpaceOrAlloc>
    ::swap(hyb_matrix& matrix)
    {
        ell.swap(matrix.ell);
        coo.swap(matrix.coo);

        thrust::swap(num_entries, matrix.num_entries);

        matrix_shape<IndexType>::swap(matrix);
    }

} // end namespace cusp

