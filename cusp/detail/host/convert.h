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
#include <cusp/array2d.h>

#include <cusp/detail/host/conversion.h>
#include <cusp/detail/host/conversion_utils.h>

#include <algorithm>
#include <string>
#include <stdexcept>

namespace cusp
{
namespace detail
{
namespace host
{

// Host Conversion Functions
// COO <- CSR
//     <- ELL
//     <- HYB
//     <- Array
// CSR <- COO
//     <- DIA
//     <- ELL
//     <- HYB
//     <- Array
// DIA <- CSR
// ELL <- CSR
// HYB <- CSR
// Array <- COO
//       <- CSR
//       <- Array (different Orientation)

/////////
// COO //
/////////
template <typename IndexType, typename ValueType>
void convert(      cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& dst,
             const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& src)
{    cusp::detail::host::csr_to_coo(dst, src);    }

template <typename IndexType, typename ValueType>
void convert(      cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& dst,
             const cusp::ell_matrix<IndexType,ValueType,cusp::host_memory>& src)
{    cusp::detail::host::ell_to_coo(dst, src);    }

template <typename IndexType, typename ValueType>
void convert(      cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& dst,
             const cusp::hyb_matrix<IndexType,ValueType,cusp::host_memory>& src)
{    cusp::detail::host::hyb_to_coo(dst, src);    }

template <typename IndexType, typename ValueType, class Orientation>
void convert(      cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& dst,
             const cusp::array2d<ValueType,cusp::host_memory,Orientation>& src)
{    cusp::detail::host::array_to_coo(dst, src);    }

template <typename IndexType, typename ValueType, class MatrixType>
void convert(      cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& dst,
             const MatrixType& src)
{
    cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> csr;
    cusp::detail::convert(csr, src);
    cusp::detail::convert(dst, csr);
}

/////////
// CSR //
/////////
template <typename IndexType, typename ValueType>
void convert(      cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& dst,
             const cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& src)
{    cusp::detail::host::coo_to_csr(dst, src);    }

template <typename IndexType, typename ValueType>
void convert(      cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& dst,
             const cusp::dia_matrix<IndexType,ValueType,cusp::host_memory>& src)
{    cusp::detail::host::dia_to_csr(dst, src);    }

template <typename IndexType, typename ValueType>
void convert(      cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& dst,
             const cusp::ell_matrix<IndexType,ValueType,cusp::host_memory>& src)
{    cusp::detail::host::ell_to_csr(dst, src);    }

template <typename IndexType, typename ValueType>
void convert(      cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& dst,
             const cusp::hyb_matrix<IndexType,ValueType,cusp::host_memory>& src)
{    cusp::detail::host::hyb_to_csr(dst, src);    }

template <typename IndexType, typename ValueType, class Orientation>
void convert(      cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& dst,
             const cusp::array2d<ValueType,cusp::host_memory,Orientation>& src)
{    cusp::detail::host::array_to_csr(dst, src);    }

/////////
// DIA //
/////////
template <typename IndexType, typename ValueType>
void convert(      cusp::dia_matrix<IndexType,ValueType,cusp::host_memory>& dst,
             const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& src,
             const float max_fill = 3.0,
             const IndexType alignment = 32)
{
    const IndexType occupied_diagonals = cusp::detail::host::count_diagonals(src);

    const float fill_ratio = float(occupied_diagonals) * float(src.num_rows) / std::max(1.0f, float(src.num_entries));

    if (max_fill < fill_ratio)
        throw cusp::format_conversion_exception("dia_matrix fill-in would exceed maximum tolerance");

    cusp::detail::host::csr_to_dia(dst, src, alignment);
}

template <typename IndexType, typename ValueType, class MatrixType>
void convert(      cusp::dia_matrix<IndexType,ValueType,cusp::host_memory>& dst,
             const MatrixType& src)
{
    cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> csr;
    cusp::detail::convert(csr, src);
    cusp::detail::convert(dst, csr);
}

/////////
// ELL //
/////////
template <typename IndexType, typename ValueType>
void convert(      cusp::ell_matrix<IndexType,ValueType,cusp::host_memory>& dst,
             const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& src,
             const float max_fill = 3.0,
             const IndexType alignment = 32)
{
    const IndexType max_entries_per_row = cusp::detail::host::compute_max_entries_per_row(src);
    const float fill_ratio = float(max_entries_per_row) * float(src.num_rows) / std::max(1.0f, float(src.num_entries));

    if (max_fill < fill_ratio)
        throw cusp::format_conversion_exception("ell_matrix fill-in would exceed maximum tolerance");

    cusp::detail::host::csr_to_ell(dst, src, max_entries_per_row, alignment);
}

template <typename IndexType, typename ValueType, class MatrixType>
void convert(      cusp::ell_matrix<IndexType,ValueType,cusp::host_memory>& dst,
             const MatrixType& src)
{
    cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> csr;
    cusp::detail::convert(csr, src);
    cusp::detail::convert(dst, csr);
}

/////////
// HYB //
/////////
template <typename IndexType, typename ValueType>
void convert(      cusp::hyb_matrix<IndexType,ValueType,cusp::host_memory>& dst,
             const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& src,
             const float relative_speed = 3.0,  const IndexType breakeven_threshold = 4096)
{
    const IndexType num_entries_per_row = cusp::detail::host::compute_optimal_entries_per_row(src, relative_speed, breakeven_threshold);
    cusp::detail::host::csr_to_hyb(dst, src, num_entries_per_row);
}

template <typename IndexType, typename ValueType, class MatrixType>
void convert(      cusp::hyb_matrix<IndexType,ValueType,cusp::host_memory>& dst,
             const MatrixType& src)
{
    cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> csr;
    cusp::detail::convert(csr, src);
    cusp::detail::convert(dst, csr);
}

///////////
// Array //
///////////
template <typename IndexType, typename ValueType, class Orientation>
void convert(      cusp::array2d<ValueType,cusp::host_memory,Orientation>& dst,
             const cusp::csr_matrix<IndexType,ValueType,cusp::host_memory>& src)
{    cusp::detail::host::csr_to_array(dst, src);    }

template <typename IndexType, typename ValueType, class Orientation>
void convert(      cusp::array2d<ValueType,cusp::host_memory,Orientation>& dst,
             const cusp::coo_matrix<IndexType,ValueType,cusp::host_memory>& src)
{    cusp::detail::host::coo_to_array(dst, src);    }

template <typename ValueType1, class Orientation1,
          typename ValueType2, class Orientation2>
void convert(      cusp::array2d<ValueType1,cusp::host_memory,Orientation1>& dst,
             const cusp::array2d<ValueType2,cusp::host_memory,Orientation2>& src)
{    cusp::detail::host::array_to_array(dst, src);   }

template <typename ValueType, class Orientation, class MatrixType>
void convert(      cusp::array2d<ValueType,cusp::host_memory,Orientation>& dst,
             const MatrixType& src)
{
    typedef typename MatrixType::index_type IndexType;
    cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> csr;
    cusp::detail::convert(csr, src);
    cusp::detail::convert(dst, csr);
}

} // end namespace host
} // end namespace detail
} // end namespace cusp

