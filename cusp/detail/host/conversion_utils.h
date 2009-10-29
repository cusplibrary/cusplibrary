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

#include <algorithm>
#include <numeric>
#include <vector>

#include <cusp/array1d.h>
#include <cusp/csr_matrix.h>

namespace cusp
{
namespace detail
{
namespace host
{

template <typename IndexType, typename ValueType>
IndexType count_diagonals(const cusp::csr_matrix<IndexType, ValueType, cusp::host_memory> & csr)
{
    std::vector<IndexType> occupied_diagonals(csr.num_rows + csr.num_cols, char(0));

    for(IndexType i = 0; i < csr.num_rows; i++){
        for(IndexType jj = csr.row_offsets[i]; jj < csr.row_offsets[i+1]; jj++){
            IndexType j = csr.column_indices[jj];
            IndexType diagonal_offset = (csr.num_rows - i) + j; //offset shifted by + num_rows
            occupied_diagonals[diagonal_offset] = 1;
        }
    }

    return std::accumulate(occupied_diagonals.begin(), occupied_diagonals.end(), IndexType(0));
}

template <typename IndexType, typename ValueType>
IndexType compute_max_entries_per_row(const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& csr)
{
    IndexType max_entries_per_row = 0;
    for(IndexType i = 0; i < csr.num_rows; i++)
        max_entries_per_row = std::max(max_entries_per_row, csr.row_offsets[i+1] - csr.row_offsets[i]); 
    return max_entries_per_row;
}


////////////////////////////////////////////////////////////////////////////////
//! Compute Optimal Number of Columns per Row in the ELL part of the HYB format
//! Examines the distribution of nonzeros per row of the input CSR matrix to find
//! the optimal tradeoff between the ELL and COO portions of the hybrid (HYB)
//! sparse matrix format under the assumption that ELL performance is a fixed
//! multiple of COO performance.  Furthermore, since ELL performance is also
//! sensitive to the absolute number of rows (and COO is not), a threshold is
//! used to ensure that the ELL portion contains enough rows to be worthwhile.
//! The default values were chosen empirically for a GTX280.
//!
//! @param csr                  CSR matrix
//! @param relative_speed       Speed of ELL relative to COO (e.g. 2.0 -> ELL is twice as fast)
//! @param breakeven_threshold  Minimum threshold at which ELL is faster than COO
////////////////////////////////////////////////////////////////////////////////
template <typename IndexType, typename ValueType>
IndexType compute_optimal_entries_per_row(const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& csr,
                                          float relative_speed = 3.0, IndexType breakeven_threshold = 4096)
{
    // compute maximum row length
    IndexType max_cols_per_row = 0;
    for(IndexType i = 0; i < csr.num_rows; i++)
        max_cols_per_row = std::max(max_cols_per_row, csr.row_offsets[i+1] - csr.row_offsets[i]); 

    // compute distribution of nnz per row
    std::vector<IndexType> histogram(max_cols_per_row + 1, 0);
    for(IndexType i = 0; i < csr.num_rows; i++)
        histogram[csr.row_offsets[i+1] - csr.row_offsets[i]]++;

    // compute optimal ELL column size 
    IndexType num_cols_per_row = max_cols_per_row;
    for(IndexType i = 0, rows = csr.num_rows; i < max_cols_per_row; i++)
    {
        rows -= histogram[i];  //number of rows of length > i
        if(relative_speed * rows < csr.num_rows || rows < breakeven_threshold)
        {
            num_cols_per_row = i;
            break;
        }
    }

    return num_cols_per_row;
}

} // end namespace host
} // end namespace detail
} // end namespace cusp

