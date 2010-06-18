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

#include <cusp/detail/matrix_traits.h>

#include <cusp/csr_matrix.h>

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
template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::detail::csr_format_tag,
             cusp::detail::coo_format_tag)
{    cusp::detail::host::csr_to_coo(src, dst);    }

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::detail::ell_format_tag,
             cusp::detail::coo_format_tag)
{    cusp::detail::host::ell_to_coo(src, dst);    }

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::detail::hyb_format_tag,
             cusp::detail::coo_format_tag)
{    cusp::detail::host::hyb_to_coo(src, dst);    }

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::detail::array2d_format_tag,
             cusp::detail::coo_format_tag)
{    cusp::detail::host::array_to_coo(src, dst);    }

template <typename Matrix1, typename Matrix2, typename MatrixFormat1>
void convert(const Matrix1& src, Matrix2& dst,
             MatrixFormat1,
             cusp::detail::coo_format_tag)
{
    typedef typename Matrix1::index_type IndexType;
    typedef typename Matrix1::value_type ValueType;
    cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> csr;
    cusp::detail::convert(src, csr);
    cusp::detail::convert(csr, dst);
}

/////////
// CSR //
/////////
template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::detail::coo_format_tag,
             cusp::detail::csr_format_tag)
{    cusp::detail::host::coo_to_csr(src, dst);    }

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::detail::dia_format_tag,
             cusp::detail::csr_format_tag)
{    cusp::detail::host::dia_to_csr(src, dst);    }

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::detail::ell_format_tag,
             cusp::detail::csr_format_tag)
{    cusp::detail::host::ell_to_csr(src, dst);    }

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::detail::hyb_format_tag,
             cusp::detail::csr_format_tag)
{    cusp::detail::host::hyb_to_csr(src, dst);    }

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::detail::array2d_format_tag,
             cusp::detail::csr_format_tag)
{    cusp::detail::host::array_to_csr(src, dst);    }

/////////
// DIA //
/////////
template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::detail::csr_format_tag,
             cusp::detail::dia_format_tag,
             const float  max_fill  = 3.0,
             const size_t alignment = 32)
{
    const size_t occupied_diagonals = cusp::detail::host::count_diagonals(src);

    const float fill_ratio = float(occupied_diagonals) * float(src.num_rows) / std::max(1.0f, float(src.num_entries));

    if (max_fill < fill_ratio && src.num_entries > (1 << 20))
        throw cusp::format_conversion_exception("dia_matrix fill-in would exceed maximum tolerance");

    cusp::detail::host::csr_to_dia(src, dst, alignment);
}

template <typename Matrix1, typename Matrix2, typename MatrixFormat1>
void convert(const Matrix1& src, Matrix2& dst,
             MatrixFormat1,
             cusp::detail::dia_format_tag)
{
    typedef typename Matrix1::index_type IndexType;
    typedef typename Matrix1::value_type ValueType;
    cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> csr;
    cusp::detail::convert(src, csr);
    cusp::detail::convert(csr, dst);
}

/////////
// ELL //
/////////
template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::detail::csr_format_tag,
             cusp::detail::ell_format_tag,
             const float  max_fill  = 3.0,
             const size_t alignment = 32)
{
    const size_t max_entries_per_row = cusp::detail::host::compute_max_entries_per_row(src);
    const float fill_ratio = float(max_entries_per_row) * float(src.num_rows) / std::max(1.0f, float(src.num_entries));

    if (max_fill < fill_ratio && src.num_entries > (1 << 20))
        throw cusp::format_conversion_exception("ell_matrix fill-in would exceed maximum tolerance");

    cusp::detail::host::csr_to_ell(src, dst, max_entries_per_row, alignment);
}

template <typename Matrix1, typename Matrix2, typename MatrixFormat1>
void convert(const Matrix1& src, Matrix2& dst,
             MatrixFormat1,
             cusp::detail::ell_format_tag)
{
    typedef typename Matrix1::index_type IndexType;
    typedef typename Matrix1::value_type ValueType;
    cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> csr;
    cusp::detail::convert(src, csr);
    cusp::detail::convert(csr, dst);
}

/////////
// HYB //
/////////
template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::detail::csr_format_tag,
             cusp::detail::hyb_format_tag,
             const float  relative_speed      = 3.0,
             const size_t breakeven_threshold = 4096)
{
    const size_t num_entries_per_row = cusp::detail::host::compute_optimal_entries_per_row(src, relative_speed, breakeven_threshold);
    cusp::detail::host::csr_to_hyb(src, dst, num_entries_per_row);
}

template <typename Matrix1, typename Matrix2, typename MatrixFormat1>
void convert(const Matrix1& src, Matrix2& dst,
             MatrixFormat1,
             cusp::detail::hyb_format_tag)
{
    typedef typename Matrix1::index_type IndexType;
    typedef typename Matrix1::value_type ValueType;
    cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> csr;
    cusp::detail::convert(src, csr);
    cusp::detail::convert(csr, dst);
}

///////////
// Array //
///////////
template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::detail::csr_format_tag,
             cusp::detail::array2d_format_tag)
{    cusp::detail::host::csr_to_array(src, dst);    }

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::detail::coo_format_tag,
             cusp::detail::array2d_format_tag)
{    cusp::detail::host::coo_to_array(src, dst);    }

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::detail::array2d_format_tag,
             cusp::detail::array2d_format_tag)
{    cusp::detail::host::array_to_array(src, dst);   }

template <typename Matrix1, typename Matrix2, typename MatrixFormat1>
void convert(const Matrix1& src, Matrix2& dst,
             MatrixFormat1,
             cusp::detail::array2d_format_tag)
{
    typedef typename Matrix1::index_type IndexType;
    typedef typename Matrix1::value_type ValueType;
    cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> csr;
    cusp::detail::convert(src, csr);
    cusp::detail::convert(csr, dst);
}




/////////////////
// Entry Point //
/////////////////
template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst)
{
    cusp::detail::host::convert(src, dst,
            typename cusp::detail::matrix_format<Matrix1>::type(),
            typename cusp::detail::matrix_format<Matrix2>::type());
}
            
} // end namespace host
} // end namespace detail
} // end namespace cusp

