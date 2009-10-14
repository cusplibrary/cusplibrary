/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the row_offsetsache License, Version 2.0 (the "License");
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

#include <cusp/csr_matrix.h>

/*
 * The standard 5-point finite difference approximation
 * to the Laplacian operator on a regular N-by-N grid.
 */
template <typename IndexType, typename ValueType>
void laplacian_5pt(cusp::csr_matrix<IndexType,ValueType,cusp::host> &csr, const IndexType N)
{
    const IndexType num_rows = N*N;
    const IndexType num_cols = N*N;
    const IndexType num_nonzeros = 5*N*N - 4*N; 

    csr.resize(num_rows, num_cols, num_nonzeros);

    if (N == 0) return;

    IndexType nz = 0;

    csr.row_offsets[0] = 0;

    for(IndexType i = 0; i < N; i++){
        for(IndexType j = 0; j < N; j++){
            IndexType indx = N*i + j;

            if (i > 0){
                csr.column_indices[nz] = indx - N;
                csr.values[nz] = -1;
                nz++;
            }

            if (j > 0){
                csr.column_indices[nz] = indx - 1;
                csr.values[nz] = -1;
                nz++;
            }

            csr.column_indices[nz] = indx;
            csr.values[nz] = 4;
            nz++;

            if (j < N - 1){
                csr.column_indices[nz] = indx + 1;
                csr.values[nz] = -1;
                nz++;
            }

            if (i < N - 1){
                csr.column_indices[nz] = indx + N;
                csr.values[nz] = -1;
                nz++;
            }
            
            csr.row_offsets[indx + 1] = nz;
        }
    }

}


