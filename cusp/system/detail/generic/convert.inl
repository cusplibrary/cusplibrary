/*
 *  Copyright 2008-2014 NVIDIA Corporation
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

#include <cusp/format.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>

#include <cusp/detail/format_utils.h>

#include <cusp/system/detail/generic/conversions/coo_to_other.h>
#include <cusp/system/detail/generic/conversions/csr_to_other.h>
#include <cusp/system/detail/generic/conversions/dia_to_other.h>
#include <cusp/system/detail/generic/conversions/ell_to_other.h>
#include <cusp/system/detail/generic/conversions/hyb_to_other.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

/////////
// COO //
/////////
template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void convert(thrust::execution_policy<DerivedPolicy>& exec,
             const SourceType& src,
             DestinationType& dst,
             cusp::csr_format,
             cusp::coo_format)
{
    csr_to_coo(exec, src, dst);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void convert(thrust::execution_policy<DerivedPolicy>& exec,
             const SourceType& src,
             DestinationType& dst,
             cusp::ell_format,
             cusp::coo_format)
{
    ell_to_coo(exec, src, dst);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void convert(thrust::execution_policy<DerivedPolicy>& exec,
             const SourceType& src,
             DestinationType& dst,
             cusp::dia_format,
             cusp::coo_format)
{
    dia_to_coo(exec, src, dst);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void convert(thrust::execution_policy<DerivedPolicy>& exec,
             const SourceType& src,
             DestinationType& dst,
             cusp::hyb_format,
             cusp::coo_format)
{
    hyb_to_coo(exec, src, dst);
}

/////////
// CSR //
/////////
template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void convert(thrust::execution_policy<DerivedPolicy>& exec,
             const SourceType& src,
             DestinationType& dst,
             cusp::coo_format,
             cusp::csr_format)
{
    coo_to_csr(exec, src, dst);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void convert(thrust::execution_policy<DerivedPolicy>& exec,
             const SourceType& src,
             DestinationType& dst,
             cusp::ell_format,
             cusp::csr_format)
{
    ell_to_csr(exec, src, dst);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void convert(thrust::execution_policy<DerivedPolicy>& exec,
             const SourceType& src,
             DestinationType& dst,
             cusp::dia_format,
             cusp::csr_format)
{
    dia_to_csr(exec, src, dst);
}


/////////
// DIA //
/////////
template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void convert(thrust::execution_policy<DerivedPolicy>& exec,
             const SourceType& src,
             DestinationType& dst,
             cusp::coo_format,
             cusp::dia_format,
             const float  max_fill  = 3.0,
             const size_t alignment = 32)
{
    const size_t occupied_diagonals = cusp::count_diagonals(exec, src);

    const float threshold  = 1e6; // 1M entries
    const float size       = float(occupied_diagonals) * float(src.num_rows);
    const float fill_ratio = size / std::max(1.0f, float(src.num_entries));

    if (max_fill < fill_ratio && size > threshold)
        throw cusp::format_conversion_exception("dia_matrix fill-in would exceed maximum tolerance");

    coo_to_dia(exec, src, dst, alignment);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void convert(thrust::execution_policy<DerivedPolicy>& exec,
             const SourceType& src,
             DestinationType& dst,
             cusp::csr_format,
             cusp::dia_format,
             const float  max_fill  = 3.0,
             const size_t alignment = 32)
{
    typename SourceType::column_indices_array_type row_indices(src.num_entries);
    cusp::detail::offsets_to_indices(src.row_offsets, row_indices);

    const size_t occupied_diagonals = cusp::detail::count_diagonals(exec, src.num_rows, src.num_cols, src.num_entries, row_indices, src.column_indices);

    const float threshold  = 1e6; // 1M entries
    const float size       = float(occupied_diagonals) * float(src.num_rows);
    const float fill_ratio = size / std::max(1.0f, float(src.num_entries));

    if (max_fill < fill_ratio && size > threshold)
        throw cusp::format_conversion_exception("dia_matrix fill-in would exceed maximum tolerance");

    csr_to_dia(exec, src, dst, alignment);
}

/////////
// ELL //
/////////
template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void convert(thrust::execution_policy<DerivedPolicy>& exec,
             const SourceType& src,
             DestinationType& dst,
             cusp::coo_format,
             cusp::ell_format,
             const float  max_fill  = 3.0,
             const size_t alignment = 32)
{
    const size_t max_entries_per_row = cusp::detail::compute_max_entries_per_row(exec, src.row_offsets);

    const float threshold  = 1e6; // 1M entries
    const float size       = float(max_entries_per_row) * float(src.num_rows);
    const float fill_ratio = size / std::max(1.0f, float(src.num_entries));

    if (max_fill < fill_ratio && size > threshold)
        throw cusp::format_conversion_exception("ell_matrix fill-in would exceed maximum tolerance");

    coo_to_ell(exec, src, dst, max_entries_per_row, alignment);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void convert(thrust::execution_policy<DerivedPolicy>& exec,
             const SourceType& src,
             DestinationType& dst,
             cusp::csr_format,
             cusp::ell_format,
             const float  max_fill  = 3.0,
             const size_t alignment = 32)
{
    const size_t max_entries_per_row = cusp::detail::compute_max_entries_per_row(exec, src.row_offsets);

    const float threshold  = 1e6; // 1M entries
    const float size       = float(max_entries_per_row) * float(src.num_rows);
    const float fill_ratio = size / std::max(1.0f, float(src.num_entries));

    if (max_fill < fill_ratio && size > threshold)
        throw cusp::format_conversion_exception("ell_matrix fill-in would exceed maximum tolerance");

    csr_to_ell(exec, src, dst, max_entries_per_row, alignment);
}


/////////
// HYB //
/////////

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void convert(thrust::execution_policy<DerivedPolicy>& exec,
             const SourceType& src,
             DestinationType& dst,
             cusp::coo_format,
             cusp::hyb_format,
             const float  relative_speed      = 3.0,
             const size_t breakeven_threshold = 4096)
{
    const size_t num_entries_per_row = cusp::detail::compute_optimal_entries_per_row(exec, src.row_offsets, relative_speed, breakeven_threshold);
    coo_to_hyb(exec, src, dst, num_entries_per_row);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void convert(thrust::execution_policy<DerivedPolicy>& exec,
             const SourceType& src,
             DestinationType& dst,
             cusp::csr_format,
             cusp::hyb_format,
             const float  relative_speed      = 3.0,
             const size_t breakeven_threshold = 4096)
{
    const size_t num_entries_per_row = cusp::detail::compute_optimal_entries_per_row(exec, src.row_offsets, relative_speed, breakeven_threshold);
    csr_to_hyb(exec, src, dst, num_entries_per_row);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void convert(thrust::execution_policy<DerivedPolicy>& exec,
             const SourceType& src,
             DestinationType& dst,
             cusp::ell_format,
             cusp::hyb_format)
{
    ell_to_hyb(exec, src, dst);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void convert(thrust::execution_policy<DerivedPolicy>& exec,
             const SourceType& src,
             DestinationType& dst,
             cusp::array2d_format,
             cusp::array1d_format)
{
    if (src.num_rows == 0 && src.num_cols == 0)
    {
        dst.resize(0);
    }
    else if (src.num_cols == 1)
    {
        dst.resize(src.num_rows);

        // interpret dst as a Nx1 column matrix and copy from src
        typedef cusp::array2d_view<typename Matrix2::view, cusp::column_major> View;
        View view(src.num_rows, 1, src.num_rows, cusp::make_array1d_view(dst));

        cusp::copy(src, view);
    }
    else if (src.num_rows == 1)
    {
        dst.resize(src.num_cols);

        // interpret dst as a 1xN row matrix and copy from src
        typedef cusp::array2d_view<typename Matrix2::view, cusp::row_major> View;
        View view(1, src.num_cols, src.num_cols, cusp::make_array1d_view(dst));

        cusp::copy(exec, src, view);
    }
    else
    {
        throw cusp::format_conversion_exception("array2d to array1d conversion is only defined for row or column vectors");
    }
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void convert(thrust::execution_policy<DerivedPolicy>& exec,
             const SourceType& src,
             DestinationType& dst,
             cusp::array1d_format,
             cusp::array2d_format)
{
    // interpret src as a Nx1 column matrix and copy to dst
    cusp::copy(exec, cusp::make_array2d_view
               (src.size(), 1, src.size(),
                cusp::make_array1d_view(src),
                cusp::column_major()),
               dst);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void convert(thrust::execution_policy<DerivedPolicy>& exec,
             const SourceType& src,
             DestinationType& dst,
             sparse_format, array2d_format)
{
    typedef typename SourceType::value_type ValueType;
    typedef typename SourceType::memory_space MemorySpace;

    cusp::array2d<ValueType,MemorySpace> tmp;
    cusp::convert(exec, src, tmp);
    cusp::convert(exec, tmp, dst);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
void convert(thrust::execution_policy<DerivedPolicy>& exec,
             const SourceType& src, DestinationType& dst,
             array2d_format, sparse_format)
{
    typedef typename SourceType::value_type ValueType;
    typedef typename SourceType::memory_space MemorySpace;

    cusp::array2d<ValueType,MemorySpace> tmp;
    cusp::convert(exec, src, tmp);
    cusp::convert(exec, tmp, dst);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp
