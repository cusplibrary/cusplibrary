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

#include <cusp/copy.h>
#include <cusp/format.h>

#include <cusp/detail/format_utils.h>

namespace cusp
{
template <typename T1,typename T2> void copy(const T1&, T2&);
template <typename P,typename T1,typename T2> void copy(const P&, const T1&, T2&);
}

#include <cusp/system/detail/generic/conversions/array_to_other.h>
#include <cusp/system/detail/generic/conversions/coo_to_other.h>
#include <cusp/system/detail/generic/conversions/csr_to_other.h>
#include <cusp/system/detail/generic/conversions/dia_to_other.h>
#include <cusp/system/detail/generic/conversions/ell_to_other.h>
#include <cusp/system/detail/generic/conversions/hyb_to_other.h>
#include <cusp/system/detail/generic/conversions/permutation_to_other.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
typename enable_if_same_system<SourceType,DestinationType>::type
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::known_format&,
        cusp::dia_format& format2)
{
    typedef typename SourceType::format Format1;

    Format1 format1;

    const size_t occupied_diagonals = cusp::detail::count_diagonals(exec, src);

    const size_t alignment = 32;
    const float max_fill   = 3.0;
    const float threshold  = 1e6; // 1M entries
    const float size       = float(occupied_diagonals) * float(src.num_rows);
    const float fill_ratio = size / std::max(1.0f, float(src.num_entries));

    if (max_fill < fill_ratio && size > threshold)
        throw cusp::format_conversion_exception("dia_matrix fill-in would exceed maximum tolerance");

    cusp::system::detail::generic::convert(exec, src, dst, format1, format2, alignment);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
typename enable_if_same_system<SourceType,DestinationType>::type
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::known_format&,
        cusp::ell_format& format2)
{
    typedef typename SourceType::format Format1;

    Format1 format1;

    const size_t max_entries_per_row = cusp::detail::compute_max_entries_per_row(exec, src);

    const size_t alignment = 32;
    const float  max_fill  = 3.0;
    const float  threshold  = 1e6; // 1M entries
    const float  size       = float(max_entries_per_row) * float(src.num_rows);
    const float  fill_ratio = size / std::max(1.0f, float(src.num_entries));

    if (max_fill < fill_ratio && size > threshold)
        throw cusp::format_conversion_exception("ell_matrix fill-in would exceed maximum tolerance");

    cusp::system::detail::generic::convert(exec, src, dst, format1, format2, max_entries_per_row, alignment);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
typename enable_if_same_system<SourceType,DestinationType>::type
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::known_format&,
        cusp::hyb_format& format2)
{
    typedef typename SourceType::format Format1;

    Format1 format1;

    const float  relative_speed      = 3.0;
    const size_t breakeven_threshold = 4096;
    const size_t alignment           = 32;

    const size_t num_entries_per_row
        = cusp::detail::compute_optimal_entries_per_row(exec, src.row_offsets, relative_speed, breakeven_threshold);

    cusp::system::detail::generic::convert(exec, src, dst, format1, format2, num_entries_per_row, alignment);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
typename enable_if_same_system<SourceType,DestinationType>::type
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        known_format&,
        known_format&)
{
    // convert src -> coo_matrix -> dst
    typename cusp::detail::as_coo_type<SourceType>::type tmp;

    if(thrust::detail::is_same<typename SourceType::format, typename DestinationType::format>::value)
    {
        cusp::copy(exec, src, dst);
    }
    else
    {
        cusp::convert(exec, src, tmp);
        cusp::convert(exec, tmp, dst);
    }
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
typename enable_if_different_system<SourceType,DestinationType>::type
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        known_format&,
        known_format&)
{
    typename cusp::detail::as_coo_type<SourceType>::type      src_tmp;
    typename cusp::detail::as_coo_type<DestinationType>::type dst_tmp;

    cusp::convert(src, src_tmp);
    cusp::copy(src_tmp, dst_tmp);
    cusp::convert(dst_tmp, dst);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp
