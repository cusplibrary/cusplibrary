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
        known_format&,
        known_format&)
{
    if(thrust::detail::is_same<typename SourceType::format, typename DestinationType::format>::value)
    {
        cusp::copy(exec, src, dst);
    }
    else
    {
        // convert src -> coo_matrix -> dst
        typename cusp::detail::as_coo_type<SourceType>::type tmp;

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
    typedef typename cusp::detail::as_matrix_type<SourceType,typename DestinationType::format>::type SrcDestType;

    SrcDestType tmp;

    cusp::convert(src, tmp);
    cusp::copy(tmp, dst);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp
