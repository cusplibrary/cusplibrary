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

#include <cusp/array1d.h>

namespace cusp
{
namespace detail
{
namespace host
{

template <typename IndexType,
          typename Array1, typename Array2,
          typename Array3, typename Array4>
IndexType spmm_csr_pass1(const IndexType num_rows, const IndexType num_cols,
                         const Array1& A_row_offsets, const Array2& A_column_indices,
                         const Array3& B_row_offsets, const Array4& B_column_indices)
{
    // Compute nnz in C (including explicit zeros)

    IndexType num_nonzeros = 0;

    cusp::array1d<IndexType,cusp::host_memory> mask(num_cols, -1);

    for(IndexType i = 0; i < num_rows; i++)
    {
        for(IndexType jj = A_row_offsets[i]; jj < A_row_offsets[i+1]; jj++)
        {
            IndexType j = A_column_indices[jj];

            for(IndexType kk = B_row_offsets[j]; kk < B_row_offsets[j+1]; kk++)
            {
                IndexType k = B_column_indices[kk];

                if(mask[k] != i)
                {
                    mask[k] = i;                        
                    num_nonzeros++;
                }
            }
        }
    }

    return num_nonzeros;
}

template <typename IndexType,
          typename Array1, typename Array2, typename Array3,
          typename Array4, typename Array5, typename Array6,
          typename Array7, typename Array8, typename Array9>
void spmm_csr_pass2(const IndexType num_rows, const IndexType num_cols,
                    const Array1& A_row_offsets, const Array2& A_column_indices, const Array3& A_values,
                    const Array4& B_row_offsets, const Array5& B_column_indices, const Array6& B_values,
                          Array7& C_row_offsets,       Array8& C_column_indices,       Array9& C_values)
{
    typedef typename Array9::value_type ValueType;

    IndexType num_nonzeros = 0;

    // Compute entries of C
    cusp::array1d<IndexType,cusp::host_memory> next(num_cols, -1);
    cusp::array1d<ValueType,cusp::host_memory> sums(num_cols,  0);
    
    num_nonzeros = 0;
    
    C_row_offsets[0] = 0;
    
    for(IndexType i = 0; i < num_rows; i++)
    {
        IndexType head   = -2;
        IndexType length =  0;
    
        IndexType jj_start = A_row_offsets[i];
        IndexType jj_end   = A_row_offsets[i+1];

        for(IndexType jj = jj_start; jj < jj_end; jj++)
        {
            IndexType j = A_column_indices[jj];
            ValueType v = A_values[jj];
    
            IndexType kk_start = B_row_offsets[j];
            IndexType kk_end   = B_row_offsets[j+1];

            for(IndexType kk = kk_start; kk < kk_end; kk++)
            {
                IndexType k = B_column_indices[kk];
    
                sums[k] += v * B_values[kk];
    
                if(next[k] == -1)
                {
                    next[k] = head;                        
                    head  = k;
                    length++;
                }
            }
        }         
    
        for(IndexType jj = 0; jj < length; jj++){
    
            if(sums[head] != 0)
            {
                C_column_indices[num_nonzeros] = head;
                C_values[num_nonzeros]         = sums[head];
                num_nonzeros++;
            }
    
            IndexType temp = head;                
            head = next[head];
    
            next[temp] = -1; //clear arrays
            sums[temp] =  0;                              
        }
    
        C_row_offsets[i+1] = num_nonzeros;
    }
    
    // XXX note: entries of C are unsorted within each row
}

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void spmm_csr(const Matrix1& A,
              const Matrix2& B,
                    Matrix3& C)
{
    typedef typename Matrix3::index_type IndexType;

    IndexType num_nonzeros = 
        spmm_csr_pass1(A.num_rows, B.num_cols,
                       A.row_offsets, A.column_indices,
                       B.row_offsets, B.column_indices);
                         
    // Resize output
    C.resize(A.num_rows, B.num_cols, num_nonzeros);
    
    spmm_csr_pass2(A.num_rows, B.num_cols,
                   A.row_offsets, A.column_indices, A.values,
                   B.row_offsets, B.column_indices, B.values,
                   C.row_offsets, C.column_indices, C.values);
}

} // end namespace host
} // end namespace detail
} // end namespace cusp

