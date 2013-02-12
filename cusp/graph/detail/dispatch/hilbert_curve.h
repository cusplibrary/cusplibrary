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
    
#include <cusp/graph/detail/host/hilbert_curve.h>
#include <cusp/graph/detail/device/hilbert_curve.h>

namespace cusp
{
namespace graph
{
namespace detail
{
namespace dispatch
{

////////////////
// Host Paths //
////////////////
template <class Array2d, class Array1d>
void hilbert_curve(const Array2d& coord, const size_t num_parts, Array1d& parts,
	    cusp::host_memory)
{
    return cusp::graph::detail::host::hilbert_curve(coord, num_parts, parts);
}

//////////////////
// Device Paths //
//////////////////
template <class Array2d, class Array1d>
void hilbert_curve(const Array2d& coord, const size_t num_parts, Array1d& parts,
	    cusp::device_memory)
{
    return cusp::graph::detail::device::hilbert_curve(coord, num_parts, parts);
}

} // end namespace dispatch
} // end namespace detail
} // end namespace graph
} // end namespace cusp

