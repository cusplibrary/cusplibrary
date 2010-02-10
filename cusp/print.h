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

#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>

#include <iostream>
#include <iomanip>

namespace cusp
{

template <typename IndexType, typename ValueType>
void print_matrix(const cusp::coo_matrix<IndexType, ValueType, cusp::host_memory>& coo)
{
    std::cout << "coo_matrix <" << coo.num_rows << ", " << coo.num_cols << "> with " << coo.num_entries << " entries\n";

    for(IndexType n = 0; n < coo.num_entries; n++)
    {
        std::cout << " " << std::setw(12) << coo.row_indices[n];
        std::cout << " " << std::setw(12) << coo.column_indices[n];
        std::cout << " " << std::setw(12) << coo.values[n] << "\n";
    }
}

template <typename IndexType, typename ValueType>
void print_matrix(const cusp::csr_matrix<IndexType, ValueType, cusp::host_memory>& csr)
{
    std::cout << "csr_matrix <" << csr.num_rows << ", " << csr.num_cols << "> with " << csr.num_entries << " entries\n";

    for(IndexType i = 0; i < csr.num_rows; i++)
    {
        for(IndexType jj = csr.row_offsets[i]; jj < csr.row_offsets[i+1]; jj++){
            std::cout << " " << std::setw(12) << i;
            std::cout << " " << std::setw(12) << csr.column_indices[jj];
            std::cout << " " << std::setw(12) << csr.values[jj] << "\n";
        }
    }
}

template <typename ValueType, typename MemorySpace, typename Orientation>
void print_matrix(const cusp::array2d<ValueType, MemorySpace, Orientation>& dense)
{
    std::cout << "array2d <" << dense.num_rows << ", " << dense.num_cols << ">\n";

    for(size_t i = 0; i < dense.num_rows; i++)
    {
        for(size_t j = 0; j < dense.num_cols; j++)
        {
            std::cout << std::setw(12) << dense(i,j);
        }

        std::cout << "\n";
    }
}

template <typename ValueType, typename MemorySpace>
void print_matrix(const cusp::array1d<ValueType, MemorySpace>& dense)
{
    std::cout << "array1d <" << dense.size() << ">\n";

    for(size_t i = 0; i < dense.size(); i++)
        std::cout << std::setw(12) << dense[i];

    std::cout << "\n";
}

template <typename MatrixType>
void print_matrix(const MatrixType& matrix)
{
    cusp::coo_matrix<int, float, cusp::host_memory> coo(matrix);
    print_matrix(coo);
}

} // end namespace cusp

