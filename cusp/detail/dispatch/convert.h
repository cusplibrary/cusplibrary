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
    std::cout << "host to device conversion" << std::endl;
}

/////////////////////////
// Device to Host Path //
/////////////////////////
template <class DestinationType, class SourceType>
void convert(DestinationType& dst, const SourceType& src, cusp::host, cusp::device)
{
    std::cout << "device to host conversion" << std::endl;
}

///////////////////////////
// Device to Device Path //
///////////////////////////
template <class DestinationType, class SourceType>
void convert(DestinationType& dst, const SourceType& src, cusp::device, cusp::device)
{
    std::cout << "device to device conversion" << std::endl;
}

} // end namespace dispatch
} // end namespace detail
} // end namespace cusp

