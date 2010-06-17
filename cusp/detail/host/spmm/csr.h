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

#include <cusp/array1d.h>

namespace cusp
{
namespace detail
{
namespace host
{

template <typename Matrix1,
          typename Matrix2,
          typename Matrix3>
void spmm_csr(const Matrix1& A,
              const Matrix2& B,
                    Matrix3& C)
{
    typedef typename Matrix3::index_type   IndexType;
    typedef typename Matrix3::value_type   ValueType;
    typedef typename Matrix3::memory_space MemorySpace;

    IndexType num_rows     = A.num_rows;
    IndexType num_cols     = B.num_cols;
    IndexType num_nonzeros = 0;

    // Compute nnz in C (including explicit zeros)
    {
        cusp::array1d<IndexType,cusp::host_memory> mask(num_cols, -1);

        for(IndexType i = 0; i < num_rows; i++)
        {
            for(IndexType jj = A.row_offsets[i]; jj < A.row_offsets[i+1]; jj++)
            {
                IndexType j = A.column_indices[jj];

                for(IndexType kk = B.row_offsets[j]; kk < B.row_offsets[j+1]; kk++)
                {
                    IndexType k = B.column_indices[kk];

                    if(mask[k] != i)
                    {
                        mask[k] = i;                        
                        num_nonzeros++;
                    }
                }
            }
        }
    }


    // Resize output
    C.resize(num_rows, num_cols, num_nonzeros);

    // Compute entries of C
    {
        cusp::array1d<IndexType,cusp::host_memory> next(num_cols, -1);
        cusp::array1d<ValueType,cusp::host_memory> sums(num_cols,  0);
        
        num_nonzeros = 0;
    
        C.row_offsets[0] = 0;
    
        for(IndexType i = 0; i < num_rows; i++)
        {
            IndexType head   = -2;
            IndexType length =  0;
    
            IndexType jj_start = A.row_offsets[i];
            IndexType jj_end   = A.row_offsets[i+1];

            for(IndexType jj = jj_start; jj < jj_end; jj++)
            {
                IndexType j = A.column_indices[jj];
                ValueType v = A.values[jj];
    
                IndexType kk_start = B.row_offsets[j];
                IndexType kk_end   = B.row_offsets[j+1];

                for(IndexType kk = kk_start; kk < kk_end; kk++)
                {
                    IndexType k = B.column_indices[kk];
    
                    sums[k] += v * B.values[kk];
    
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
                    C.column_indices[num_nonzeros] = head;
                    C.values[num_nonzeros]         = sums[head];
                    num_nonzeros++;
                }
    
                IndexType temp = head;                
                head = next[head];
    
                next[temp] = -1; //clear arrays
                sums[temp] =  0;                              
            }
    
            C.row_offsets[i+1] = num_nonzeros;
        }
    }

    // entries of C are unsorted
}

} // end namespace host
} // end namespace detail
} // end namespace cusp

