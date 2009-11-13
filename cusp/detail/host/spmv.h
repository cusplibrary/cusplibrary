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

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

namespace cusp
{
namespace detail
{
namespace host
{

// coo_matrix
template <typename IndexType, typename ValueType>
void spmv_coo(const IndexType num_rows,
              const IndexType num_cols,
              const IndexType num_entries,
	          const IndexType * row_indices, 
	          const IndexType * column_indices, 
	          const ValueType * values,
	          const ValueType * x,
	                      ValueType * y)
{
    for(IndexType n = 0; n < num_entries; n++)
    {
        y[row_indices[n]] += values[n] * x[column_indices[n]];
    }
}


template <typename IndexType, typename ValueType>
void spmv(const cusp::coo_matrix<IndexType, ValueType, cusp::host_memory>& coo, 
          const ValueType * x,  
                ValueType * y)
{
    spmv_coo(coo.num_rows, coo.num_cols, coo.num_entries,
             &coo.row_indices[0], &coo.column_indices[0], &coo.values[0],
             x, y);
}

// csr_matrix
template <typename IndexType, typename ValueType>
void spmv_csr(const IndexType num_rows, 
              const IndexType num_cols,
              const IndexType * row_offsets, 
              const IndexType * column_indices, 
              const ValueType * values, 
              const ValueType * x,    
                    ValueType * y)    
{
    for (IndexType i = 0; i < num_rows; i++)
    {
        const IndexType row_start = row_offsets[i];
        const IndexType row_end   = row_offsets[i+1];

        ValueType sum = y[i];

        for (IndexType jj = row_start; jj < row_end; jj++)
        {
            const IndexType j = column_indices[jj];  //column index
            sum += values[jj] * x[j];
        }

        y[i] = sum; 
    }
}

template <typename IndexType, typename ValueType>
void spmv(const cusp::csr_matrix<IndexType, ValueType, cusp::host_memory>& csr, 
          const ValueType * x,  
                ValueType * y)
{
    spmv_csr(csr.num_rows, csr.num_cols,
             &csr.row_offsets[0], &csr.column_indices[0], &csr.values[0],
             x, y);
}


// dia_matrix
template <typename IndexType, typename OffsetType, typename ValueType>
void spmv_dia(const IndexType num_rows,
              const IndexType num_cols,
              const IndexType num_diagonals,
              const IndexType stride,
              const OffsetType * diagonal_offsets, 
              const ValueType  * values, 
              const ValueType  * x,
                    ValueType  * y)
{
    for(IndexType i = 0; i < num_diagonals; i++)
    {
        const OffsetType k = diagonal_offsets[i];  //diagonal offset

        const IndexType i_start = std::max((OffsetType) 0, -k);
        const IndexType j_start = std::max((OffsetType) 0,  k);

        //number of elements to process
        const IndexType N = std::min(num_rows - i_start, num_cols - j_start);

        const ValueType * d_ = values + i*stride + i_start;
        const ValueType * x_ = x + j_start;
              ValueType * y_ = y + i_start;

        for(IndexType n = 0; n < N; n++)
            y_[n] += d_[n] * x_[n]; 
    }
}

template <typename IndexType, typename ValueType>
void spmv(const dia_matrix<IndexType, ValueType, cusp::host_memory>& dia, 
          const ValueType * x,  
                ValueType * y)
{
    IndexType num_diagonals = dia.values.num_cols;
    IndexType stride        = dia.values.num_rows;

    spmv_dia(dia.num_rows, dia.num_cols, num_diagonals, stride,
             &dia.diagonal_offsets[0], &dia.values.values[0], 
             x, y);
}


// ell_matrix
template <typename IndexType, typename ValueType>
void spmv_ell(const IndexType num_rows,
              const IndexType num_cols,
              const IndexType num_entries_per_row,
              const IndexType stride,
              const IndexType * column_indices, 
              const ValueType * values, 
              const ValueType * x,
                    ValueType * y)
{
    const IndexType invalid_index = cusp::ell_matrix<IndexType, ValueType, cusp::host_memory>::invalid_index;

    for(IndexType n = 0; n < num_entries_per_row; n++)
    {
        const IndexType * Aj_n = column_indices + n * stride;
        const ValueType * Ax_n = values         + n * stride;

        for(IndexType i = 0; i < num_rows; i++)
        {
            if (Aj_n[i] != invalid_index)
                y[i] += Ax_n[i] * x[Aj_n[i]];
        }
    }
}

template <typename IndexType, typename ValueType>
void spmv(const cusp::ell_matrix<IndexType, ValueType, cusp::host_memory>& ell, 
          const ValueType * x,  
                ValueType * y)
{
    const IndexType stride              = ell.column_indices.num_rows;
    const IndexType num_entries_per_row = ell.column_indices.num_cols;

    spmv_ell(ell.num_rows, ell.num_cols, num_entries_per_row, stride,
             &ell.column_indices.values[0], &ell.values.values[0],
             x, y);
}


// hyb_matrix
template <typename IndexType, typename ValueType>
void spmv(const cusp::hyb_matrix<IndexType, ValueType, cusp::host_memory>& hyb, 
          const ValueType * x,  
                ValueType * y)
{
    spmv(hyb.ell, x, y);
    spmv(hyb.coo, x, y);
}


} // end namespace host
} // end namespace detail
} // end namespace cusp

