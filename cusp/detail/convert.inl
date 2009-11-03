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
    
#include <cusp/detail/dispatch/convert.h>

namespace cusp
{
namespace detail
{

template <class DestinationType, class SourceType>
void convert(DestinationType& dst, const SourceType& src)
{
    typedef typename DestinationType::memory_space destination_space;
    typedef typename SourceType::memory_space      source_space;

    cusp::detail::dispatch::convert(dst, src, destination_space(), source_space());
}

} // end namespace detail
} // end namespace cusp

