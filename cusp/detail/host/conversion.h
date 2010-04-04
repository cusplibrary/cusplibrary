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

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/array2d.h>

#include <cusp/exception.h>

#include <cusp/detail/host/conversion_utils.h>

namespace cusp
{
namespace detail
{
namespace host
{

/////////////////////
// COO Conversions //
/////////////////////
    
template <typename IndexType, typename ValueType>
void coo_to_csr(      cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& dst,
                const cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& src)
{
    dst.resize(src.num_rows, src.num_cols, src.num_entries);
    
    //compute number of non-zero entries per row of A 
    std::fill(dst.row_offsets.begin(), dst.row_offsets.end(), IndexType(0));

    for (IndexType n = 0; n < src.num_entries; n++)
        dst.row_offsets[src.row_indices[n]]++;

    //cumsum the num_entries per row to get dst.row_offsets[]
    for(IndexType i = 0, cumsum = 0; i < src.num_rows; i++)
    {
        IndexType temp = dst.row_offsets[i];
        dst.row_offsets[i] = cumsum;
        cumsum += temp;
    }
    dst.row_offsets[src.num_rows] = src.num_entries; 

    //write Aj,Ax into dst.column_indices,dst.values
    for(IndexType n = 0; n < src.num_entries; n++)
    {
        IndexType row  = src.row_indices[n];
        IndexType dest = dst.row_offsets[row];

        dst.column_indices[dest] = src.column_indices[n];
        dst.values[dest]         = src.values[n];

        dst.row_offsets[row]++;
    }

    for(IndexType i = 0, last = 0; i <= src.num_rows; i++)
    {
        IndexType temp = dst.row_offsets[i];
        dst.row_offsets[i]  = last;
        last   = temp;
    }

    //csr may contain duplicates
}

template <typename IndexType, typename ValueType, class Orientation>
void coo_to_array(      cusp::array2d<ValueType,cusp::host_memory,Orientation>& dst,
                  const cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& src)
{
    dst.resize(src.num_rows, src.num_cols);

    std::fill(dst.values.begin(), dst.values.end(), ValueType(0));

    for(IndexType n = 0; n < src.num_entries; n++)
        dst(src.row_indices[n], src.column_indices[n]) += src.values[n]; //sum duplicates
}

/////////////////////
// CSR Conversions //
/////////////////////

template <typename IndexType, typename ValueType>
void csr_to_coo(      cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& dst,
                const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& src)
{
    dst.resize(src.num_rows, src.num_cols, src.num_entries);
    
    for(IndexType i = 0; i < src.num_rows; i++)
        for(IndexType jj = src.row_offsets[i]; jj < src.row_offsets[i + 1]; jj++)
            dst.row_indices[jj] = i;

    dst.column_indices = src.column_indices;
    dst.values         = src.values;
}

template <typename IndexType, typename ValueType>
void csr_to_dia(       cusp::dia_matrix<IndexType,ValueType,cusp::host_memory>& dia,
                 const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& csr,
                 const IndexType alignment = 32)
{
    // compute number of occupied diagonals and enumerate them
    IndexType num_diagonals = 0;

    cusp::array1d<IndexType,cusp::host_memory> diag_map(csr.num_rows + csr.num_cols, 0);

    for(IndexType i = 0; i < csr.num_rows; i++)
    {
        for(IndexType jj = csr.row_offsets[i]; jj < csr.row_offsets[i+1]; jj++)
        {
            IndexType j = csr.column_indices[jj];
            IndexType map_index = (csr.num_rows - i) + j; //offset shifted by + num_rows
            if(diag_map[map_index] == 0)
            {
                diag_map[map_index] = 1;
                num_diagonals++;
            }
        }
    }
   

    // allocate DIA structure
    dia.resize(csr.num_rows, csr.num_cols, csr.num_entries, num_diagonals, alignment);

    // fill in diagonal_offsets array
    for(IndexType n = 0, diag = 0; n < csr.num_rows + csr.num_cols; n++)
    {
        if(diag_map[n] == 1)
        {
            diag_map[n] = diag;
            dia.diagonal_offsets[diag] = (IndexType) n - (IndexType) csr.num_rows;
            diag++;
        }
    }

    // fill in values array
    std::fill(dia.values.values.begin(), dia.values.values.end(), ValueType(0));

    for(IndexType i = 0; i < csr.num_rows; i++)
    {
        for(IndexType jj = csr.row_offsets[i]; jj < csr.row_offsets[i+1]; jj++)
        {
            IndexType j = csr.column_indices[jj];
            IndexType map_index = (csr.num_rows - i) + j; //offset shifted by + num_rows
            IndexType diag = diag_map[map_index];
        
            dia.values(i, diag) = csr.values[jj];
        }
    }
}
    

template <typename IndexType, typename ValueType>
void csr_to_hyb(      cusp::hyb_matrix<IndexType,ValueType,cusp::host_memory>& hyb, 
                const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& csr,
                const IndexType num_entries_per_row,
                const IndexType alignment = 32)
{
    // The ELL portion of the HYB matrix will have 'num_entries_per_row' columns.
    // Nonzero values that do not fit within the ELL structure are placed in the 
    // COO format portion of the HYB matrix.
    
    cusp::ell_matrix<IndexType, ValueType, cusp::host_memory> & ell = hyb.ell;
    cusp::coo_matrix<IndexType, ValueType, cusp::host_memory> & coo = hyb.coo;

    // compute number of nonzeros in the ELL and COO portions
    IndexType num_ell_entries = 0;
    for(IndexType i = 0; i < csr.num_rows; i++)
        num_ell_entries += std::min(num_entries_per_row, csr.row_offsets[i+1] - csr.row_offsets[i]); 

    IndexType num_coo_entries = csr.num_entries - num_ell_entries;

    hyb.resize(csr.num_rows, csr.num_cols, 
               num_ell_entries, num_coo_entries, 
               num_entries_per_row, alignment);

    const IndexType invalid_index = cusp::ell_matrix<IndexType, ValueType, cusp::host_memory>::invalid_index;

    // pad out ELL format with zeros
    std::fill(ell.column_indices.values.begin(), ell.column_indices.values.end(), invalid_index);
    std::fill(ell.values.values.begin(),         ell.values.values.end(),         ValueType(0));

    for(IndexType i = 0, coo_nnz = 0; i < csr.num_rows; i++)
    {
        IndexType n = 0;
        IndexType jj = csr.row_offsets[i];

        // copy up to num_cols_per_row values of row i into the ELL
        while(jj < csr.row_offsets[i+1] && n < num_entries_per_row)
        {
            ell.column_indices(i,n) = csr.column_indices[jj];
            ell.values(i,n)         = csr.values[jj];
            jj++, n++;
        }

        // copy any remaining values in row i into the COO
        while(jj < csr.row_offsets[i+1])
        {
            coo.row_indices[coo_nnz]    = i;
            coo.column_indices[coo_nnz] = csr.column_indices[jj];
            coo.values[coo_nnz]         = csr.values[jj];
            jj++; coo_nnz++;
        }
    }
}


template <typename IndexType, typename ValueType>
void csr_to_ell(      cusp::ell_matrix<IndexType,ValueType,cusp::host_memory>&  ell,
                const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>&  csr,
                const IndexType num_entries_per_row, const IndexType alignment = 32)
{
    // Constructs an ELL matrix with 'num_entries_per_row' consisting of the first
    // 'num_entries_per_row' entries in each row of the CSR matrix.
    cusp::hyb_matrix<IndexType, ValueType, cusp::host_memory> hyb;

    csr_to_hyb(hyb, csr, num_entries_per_row, alignment);

    ell.swap(hyb.ell);
}

    
template <typename IndexType, typename ValueType, class Orientation>
void csr_to_array(      cusp::array2d<ValueType,cusp::host_memory,Orientation>& dst,
                  const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& src)
{
    dst.resize(src.num_rows, src.num_cols);

    std::fill(dst.values.begin(), dst.values.end(), ValueType(0));

    for(IndexType i = 0; i < src.num_rows; i++)
        for(IndexType jj = src.row_offsets[i]; jj < src.row_offsets[i+1]; jj++)
            dst(i, src.column_indices[jj]) += src.values[jj]; //sum duplicates
}


/////////////////////
// DIA Conversions //
/////////////////////

template <typename IndexType, typename ValueType>
void dia_to_csr(      cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& dst,
                const cusp::dia_matrix<IndexType,ValueType,cusp::host_memory>& src)
{
    IndexType num_entries = 0;
    IndexType num_diagonals = src.diagonal_offsets.size();
    
    // count nonzero entries
    for(IndexType i = 0; i < src.num_rows; i++)
    {
        for(IndexType n = 0; n < num_diagonals; n++)
        {
            const IndexType j = i + src.diagonal_offsets[n];

            if(j >= 0 && j < src.num_cols && src.values(i,n) != 0)
                num_entries++;
        }
    }

    dst.resize(src.num_rows, src.num_cols, num_entries);

    num_entries = 0;
    dst.row_offsets[0] = 0;

    // copy nonzero entries to CSR structure
    for(IndexType i = 0; i < src.num_rows; i++)
    {
        for(IndexType n = 0; n < num_diagonals; n++)
        {
            const IndexType j = i + src.diagonal_offsets[n];

            if(j >= 0 && j < src.num_cols)
            {
                const ValueType value = src.values(i, n);

                if (value != 0)
                {
                    dst.column_indices[num_entries] = j;
                    dst.values[num_entries] = value;
                    num_entries++;
                }
            }
        }

        dst.row_offsets[i + 1] = num_entries;
    }
}

/////////////////////
// ELL Conversions //
/////////////////////

template <typename IndexType, typename ValueType>
void ell_to_coo(      cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& dst,
                const cusp::ell_matrix<IndexType,ValueType,cusp::host_memory>& src)
{
    const IndexType invalid_index = cusp::ell_matrix<IndexType, ValueType, cusp::host_memory>::invalid_index;
    
    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    IndexType num_entries = 0;

    const IndexType num_entries_per_row = src.column_indices.num_cols;

    for(IndexType i = 0; i < src.num_rows; i++)
    {
        for(IndexType n = 0; n < num_entries_per_row; n++)
        {
            const IndexType j = src.column_indices(i,n);
            const ValueType v = src.values(i,n);

            if(j != invalid_index)
            {
                dst.row_indices[num_entries]    = i;
                dst.column_indices[num_entries] = j;
                dst.values[num_entries]         = v;
                num_entries++;
            }
        }
    }
}

template <typename IndexType, typename ValueType>
void ell_to_csr(      cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& dst,
                const cusp::ell_matrix<IndexType,ValueType,cusp::host_memory>& src)
{
    const IndexType invalid_index = cusp::ell_matrix<IndexType, ValueType, cusp::host_memory>::invalid_index;

    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    IndexType num_entries = 0;
    dst.row_offsets[0] = 0;

    const IndexType num_entries_per_row = src.column_indices.num_cols;

    for(IndexType i = 0; i < src.num_rows; i++)
    {
        for(IndexType n = 0; n < num_entries_per_row; n++)
        {
            const IndexType j = src.column_indices(i,n);
            const ValueType v = src.values(i,n);

            if(j != invalid_index)
            {
                dst.column_indices[num_entries] = j;
                dst.values[num_entries]         = v;
                num_entries++;
            }
        }

        dst.row_offsets[i + 1] = num_entries;
    }
}

/////////////////////
// HYB Conversions //
/////////////////////

template <typename IndexType, typename ValueType>
void hyb_to_coo(      cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& dst,
                const cusp::hyb_matrix<IndexType,ValueType,cusp::host_memory>& src)
{
    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    const cusp::ell_matrix<IndexType,ValueType,cusp::host_memory>& ell = src.ell;
    const cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& coo = src.coo;

    const IndexType invalid_index = cusp::ell_matrix<IndexType, ValueType, cusp::host_memory>::invalid_index;
    const IndexType num_entries_per_row = ell.column_indices.num_cols;

    IndexType num_entries  = 0;
    IndexType coo_progress = 0;
    
    // merge each row of the ELL and COO parts into a single COO row
    for(IndexType i = 0; i < src.num_rows; i++)
    {
        // append the i-th row from the ELL part
        for(IndexType n = 0; n < num_entries_per_row; n++)
        {
            const IndexType j = ell.column_indices(i,n);
            const ValueType v = ell.values(i,n);

            if(j != invalid_index)
            {
                dst.row_indices[num_entries]    = i;
                dst.column_indices[num_entries] = j;
                dst.values[num_entries]         = v;
                num_entries++;
            }
        }

        // append the i-th row from the COO part
        while (coo_progress < coo.num_entries && coo.row_indices[coo_progress] == i)
        {
            dst.row_indices[num_entries]    = i;
            dst.column_indices[num_entries] = coo.column_indices[coo_progress];
            dst.values[num_entries]         = coo.values[coo_progress];
            num_entries++;
            coo_progress++;
        }
    }
}

template <typename IndexType, typename ValueType>
void hyb_to_csr(      cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& dst,
                const cusp::hyb_matrix<IndexType,ValueType,cusp::host_memory>& src)
{
    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    const cusp::ell_matrix<IndexType,ValueType,cusp::host_memory>& ell = src.ell;
    const cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& coo = src.coo;

    const IndexType invalid_index = cusp::ell_matrix<IndexType, ValueType, cusp::host_memory>::invalid_index;
    const IndexType num_entries_per_row = ell.column_indices.num_cols;

    IndexType num_entries = 0;
    dst.row_offsets[0] = 0;
    
    IndexType coo_progress = 0;
    
    // merge each row of the ELL and COO parts into a single CSR row
    for(IndexType i = 0; i < src.num_rows; i++)
    {
        // append the i-th row from the ELL part
        for(IndexType n = 0; n < num_entries_per_row; n++)
        {
            const IndexType j = ell.column_indices(i,n);
            const ValueType v = ell.values(i,n);

            if(j != invalid_index)
            {
                dst.column_indices[num_entries] = j;
                dst.values[num_entries]         = v;
                num_entries++;
            }
        }

        // append the i-th row from the COO part
        while (coo_progress < coo.num_entries && coo.row_indices[coo_progress] == i)
        {
            dst.column_indices[num_entries] = coo.column_indices[coo_progress];
            dst.values[num_entries]         = coo.values[coo_progress];
            num_entries++;
            coo_progress++;
        }

        dst.row_offsets[i + 1] = num_entries;
    }
}


///////////////////////
// Dense Conversions //
///////////////////////
template <typename ValueType1, class Orientation1,
          typename ValueType2, class Orientation2>
void array_to_array(      cusp::array2d<ValueType1,cusp::host_memory,Orientation1>& dst,
                    const cusp::array2d<ValueType2,cusp::host_memory,Orientation2>& src)
{
    dst.resize(src.num_rows, src.num_cols);

    for(size_t i = 0; i < src.num_rows; i++)
        for(size_t j = 0; j < src.num_cols; j++)
            dst(i,j) = src(i,j);
}


template <typename IndexType, typename ValueType, class Orientation>
void array_to_coo(      cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& dst,
                  const cusp::array2d<ValueType,cusp::host_memory,Orientation>& src)
{
    IndexType nnz = src.num_entries - std::count(src.values.begin(), src.values.end(), ValueType(0));

    dst.resize(src.num_rows, src.num_cols, nnz);

    nnz = 0;

    for(size_t i = 0; i < src.num_rows; i++)
    {
        for(size_t j = 0; j < src.num_cols; j++)
        {
            if (src(i,j) != 0)
            {
                dst.row_indices[nnz]    = i;
                dst.column_indices[nnz] = j;
                dst.values[nnz]         = src(i,j);
                nnz++;
            }
        }
    }
}

template <typename IndexType, typename ValueType, class Orientation>
void array_to_csr(      cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& dst,
                  const cusp::array2d<ValueType,cusp::host_memory,Orientation>& src)
{
    IndexType nnz = src.num_entries - std::count(src.values.begin(), src.values.end(), ValueType(0));

    dst.resize(src.num_rows, src.num_cols, nnz);

    IndexType num_entries = 0;

    for(size_t i = 0; i < src.num_rows; i++)
    {
        dst.row_offsets[i] = num_entries;

        for(size_t j = 0; j < src.num_cols; j++)
        {
            if (src(i,j) != 0){
                dst.column_indices[num_entries] = j;
                dst.values[num_entries]         = src(i,j);
                num_entries++;
            }
        }
    }

    dst.row_offsets[src.num_rows] = num_entries;
}

} // end namespace host
} // end namespace detail
} // end namespace cusp

