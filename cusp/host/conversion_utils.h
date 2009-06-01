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

#include <cusp/csr_matrix.h>

namespace cusp
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
//! multiple of COO performance.
//! @param csr          CSR matrix
//! @param ratio        Ratio of rows that must be occupied to add column to 
//                      the ELL portion of the matrix.  Equivalently, a rough
//                      measure of COO performance relative to ELL. For example,
//                      ratio = 0.25 implies that ELL is four times faster.
////////////////////////////////////////////////////////////////////////////////

// relative speed of full ELL vs. COO (full = no padding)
template <typename IndexType, typename ValueType>
IndexType compute_optimal_entries_per_row(const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& csr, 
                                          const float ratio)
{
    const IndexType threshold = ratio * csr.num_rows;

    // compute maximum row length
    IndexType max_entries_per_row = compute_max_entries_per_row(csr);

    // compute distribution of nnz per row
    IndexType * histogram = cusp::new_host_array<IndexType>(max_entries_per_row + 1);
    std::fill(histogram, histogram + max_entries_per_row + 1, 0);
    for(IndexType i = 0; i < csr.num_rows; i++)
        histogram[csr.row_offsets[i+1] - csr.row_offsets[i]]++;

    // compute optimal ELL column size 
    IndexType num_entries_per_row = max_entries_per_row;
    for(IndexType i = 0, rows = csr.num_rows; i < max_entries_per_row; i++){
        rows -= histogram[i];  //number of rows of length > i
        if( rows < threshold ){
            num_entries_per_row = i;
            break;
        }
    }
    cusp::delete_host_array(histogram);

    return num_entries_per_row;
}

} // end namespace host

} // end namespace cusp

