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
    
#include <cusp/vector.h>

#include <cusp/detail/host/convert.h>

namespace cusp
{
namespace detail
{
namespace dispatch
{

///////////////////////
// Host to Host Path //
///////////////////////
template <class DestinationType, class SourceType>
void convert(DestinationType& dst, const SourceType& src, cusp::host, cusp::host)
{
    cusp::detail::host::convert(dst, src);
}

/////////////////////////
// Host to Device Path //
/////////////////////////
template <class DestinationType, class SourceType>
void convert(DestinationType& dst, const SourceType& src, cusp::device, cusp::host)
{
    // convert on host and transfer to device
    typedef typename DestinationType::template rebind<cusp::host>::type HostDestinationType;
    
    HostDestinationType tmp;

    cusp::detail::host::convert(tmp, src);

    dst = tmp;
}

/////////////////////////
// Device to Host Path //
/////////////////////////
template <class DestinationType, class SourceType>
void convert(DestinationType& dst, const SourceType& src, cusp::host, cusp::device)
{
    // transfer to host and transfer to device
    typedef typename SourceType::template rebind<cusp::host>::type HostSourceType;
    
    HostSourceType tmp = src;

    cusp::detail::host::convert(dst, tmp);
}

///////////////////////////
// Device to Device Path //
///////////////////////////
template <class DestinationType, class SourceType>
void convert(DestinationType& dst, const SourceType& src, cusp::device, cusp::device)
{
    // transfer to host, convert on host, and transfer back to device
    typedef typename SourceType::template rebind<cusp::host>::type      HostSourceType;
    typedef typename DestinationType::template rebind<cusp::host>::type HostDestinationType;

    HostSourceType tmp1(src);

    HostDestinationType tmp2;

    cusp::detail::host::convert(tmp2, tmp1);

    dst = tmp2;
}

} // end namespace dispatch
} // end namespace detail
} // end namespace cusp

