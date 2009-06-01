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
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/dense_matrix.h>

#include <cusp/host/conversion.h>
#include <cusp/host/conversion_utils.h>

#include <algorithm>
#include <string>
#include <stdexcept>

namespace cusp
{

    class format_conversion_exception : public std::exception
    {
        public:
            format_conversion_exception(const std::string _msg) : msg(_msg) {}
            ~format_conversion_exception() throw() {}
            const char* what() const throw() { return msg.c_str(); }

        private:
            std::string msg;
    };


namespace detail
{


// Host Conversion Functions
// COO
//  - CSR
//  - Dense
// CSR
//  - COO
//  - DIA
//  - ELL
//  - HYB
//  - Dense
// Dense
//  - COO
//  - CSR


/////////
// COO //
/////////
template <typename IndexType, typename ValueType>
void convert_format(      cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& dst,
                    const cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& src)
{    cusp::host::coo_to_csr(dst, src);    }

template <typename IndexType, typename ValueType, class Orientation>
void convert_format(      cusp::dense_matrix<ValueType,cusp::host_memory,Orientation>& dst,
                    const cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& src)
{    cusp::host::coo_to_dense(dst, src);    }


/////////
// CSR //
/////////
template <typename IndexType, typename ValueType>
void convert_format(      cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& dst,
                    const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& src)
{    cusp::host::csr_to_coo(dst, src);    }


template <typename IndexType, typename ValueType>
void convert_format(      cusp::dia_matrix<IndexType,ValueType,cusp::host_memory>& dst,
                    const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& src,
                    const float max_fill = 3.0,
                    const IndexType alignment = 16)
{
    const IndexType occupied_diagonals = cusp::host::count_diagonals(src);

    const float fill_ratio = float(occupied_diagonals) * float(src.num_rows) / std::max(1.0f, float(src.num_entries));

    if (max_fill < fill_ratio)
        throw cusp::format_conversion_exception("dia_matrix fill-in would exceed maximum tolerance");

    cusp::host::csr_to_dia(dst, src, alignment);
}

template <typename IndexType, typename ValueType>
void convert_format(      cusp::ell_matrix<IndexType,ValueType,cusp::host_memory>& dst,
                    const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& src,
                    const float max_fill = 3.0,
                    const IndexType alignment = 16)
{
    const IndexType max_entries_per_row = cusp::host::compute_max_entries_per_row(src);
    const float fill_ratio = float(max_entries_per_row) * float(src.num_rows) / std::max(1.0f, float(src.num_entries));

    if (max_fill < fill_ratio)
        throw cusp::format_conversion_exception("ell_matrix fill-in would exceed maximum tolerance");

    cusp::host::csr_to_ell(dst, src, max_entries_per_row, alignment);
}

template <typename IndexType, typename ValueType>
void convert_format(      cusp::hyb_matrix<IndexType,ValueType,cusp::host_memory>& dst,
                    const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& src,
                    const float ratio = 0.35,
                    const IndexType alignment = 16)
{
    const IndexType num_entries_per_row = cusp::host::compute_optimal_entries_per_row(src, ratio);
    cusp::host::csr_to_hyb(dst, src, num_entries_per_row, alignment);
}

template <typename IndexType, typename ValueType, class Orientation>
void convert_format(      cusp::dense_matrix<ValueType,cusp::host_memory,Orientation>& dst,
                    const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& src)
{    cusp::host::csr_to_dense(dst, src);    }


///////////
// Dense //
///////////
template <typename IndexType, typename ValueType, class Orientation>
void convert_format(      cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& dst,
                    const cusp::dense_matrix<ValueType,cusp::host_memory,Orientation>& src)
{    cusp::host::dense_to_coo(dst, src);    }

template <typename IndexType, typename ValueType, class Orientation>
void convert_format(      cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& dst,
                    const cusp::dense_matrix<ValueType,cusp::host_memory,Orientation>& src)
{    cusp::host::dense_to_csr(dst, src);    }


//////////////////////////
// No Format Conversion //
//////////////////////////
template <typename IndexType, typename ValueType, class MemorySpace1, class MemorySpace2>
void convert_format(      cusp::coo_matrix<IndexType,ValueType,MemorySpace1>& dst,
                    const cusp::coo_matrix<IndexType,ValueType,MemorySpace2>& src)
{    cusp::allocate_matrix_like(dst, src); cusp::memcpy_matrix(dst, src);    }

template <typename IndexType, typename ValueType, class MemorySpace1, class MemorySpace2>
void convert_format(      cusp::csr_matrix<IndexType,ValueType,MemorySpace1>& dst,
                    const cusp::csr_matrix<IndexType,ValueType,MemorySpace2>& src)
{    cusp::allocate_matrix_like(dst, src); cusp::memcpy_matrix(dst, src);    }

template <typename IndexType, typename ValueType, class MemorySpace1, class MemorySpace2>
void convert_format(      cusp::dia_matrix<IndexType,ValueType,MemorySpace1>& dst,
                    const cusp::dia_matrix<IndexType,ValueType,MemorySpace2>& src)
{    cusp::allocate_matrix_like(dst, src); cusp::memcpy_matrix(dst, src);    }

template <typename IndexType, typename ValueType, class MemorySpace1, class MemorySpace2>
void convert_format(      cusp::ell_matrix<IndexType,ValueType,MemorySpace1>& dst,
                    const cusp::ell_matrix<IndexType,ValueType,MemorySpace2>& src)
{    cusp::allocate_matrix_like(dst, src); cusp::memcpy_matrix(dst, src);    }

template <typename IndexType, typename ValueType, class MemorySpace1, class MemorySpace2>
void convert_format(      cusp::hyb_matrix<IndexType,ValueType,MemorySpace1>& dst,
                    const cusp::hyb_matrix<IndexType,ValueType,MemorySpace2>& src)
{    cusp::allocate_matrix_like(dst, src); cusp::memcpy_matrix(dst, src);    }



/////////////////////
// Default Handler //
/////////////////////
//Template <class MatrixType1, class MatrixType2>
//Void convert_format(MatrixType1& dst, MatrixType2& src)
//{
//    cusp::detail::convert_format(dst, src);
//}

} // end namespace detail



// entry point
template <class MatrixType1, class MatrixType2>
void convert_matrix(MatrixType1& dst, MatrixType2& src)
{
    cusp::detail::convert_format(dst, src);
}


} // end namespace cusp
