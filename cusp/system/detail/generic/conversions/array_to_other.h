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

#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/copy.h>
#include <cusp/format.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

using namespace cusp::detail;

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
typename enable_if_same_system<SourceType,DestinationType>::type
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::array2d_format&,
        cusp::array1d_format&)
{
    if (src.num_rows == 0 && src.num_cols == 0)
    {
        dst.resize(0);
    }
    else if (src.num_cols == 1)
    {
        dst.resize(src.num_rows);

        // interpret dst as a Nx1 column matrix and copy from src
        typedef cusp::array2d_view<typename DestinationType::view, cusp::column_major> View;
        View view(src.num_rows, 1, src.num_rows, cusp::make_array1d_view(dst));

        cusp::copy(exec, src, view);
    }
    else if (src.num_rows == 1)
    {
        dst.resize(src.num_cols);

        // interpret dst as a 1xN row matrix and copy from src
        typedef cusp::array2d_view<typename DestinationType::view, cusp::row_major> View;
        View view(1, src.num_cols, src.num_cols, cusp::make_array1d_view(dst));

        cusp::copy(exec, src, view);
    }
    else
    {
        throw cusp::format_conversion_exception("array2d to array1d conversion is only defined for row or column vectors");
    }
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
typename enable_if_same_system<SourceType,DestinationType>::type
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::array1d_format&,
        cusp::array2d_format&)
{
    // interpret src as a Nx1 column matrix and copy to dst
    cusp::copy(exec, cusp::make_array2d_view
               (src.size(), 1, src.size(),
                cusp::make_array1d_view(src),
                cusp::column_major()),
               dst);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
typename enable_if_same_system<SourceType,DestinationType>::type
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::array2d_format&,
        cusp::array2d_format&)
{
    cusp::copy(exec, src, dst);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
typename enable_if_same_system<SourceType,DestinationType>::type
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        known_format&,
        array2d_format&)
{
    typedef typename SourceType::value_type ValueType;
    typedef typename SourceType::memory_space MemorySpace;

    cusp::array2d<ValueType,MemorySpace> tmp;
    cusp::convert(exec, src, tmp);
    cusp::convert(exec, tmp, dst);
}

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
typename enable_if_same_system<SourceType,DestinationType>::type
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        array2d_format&,
        known_format&)
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
