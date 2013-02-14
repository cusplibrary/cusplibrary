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
    
#include <cusp/graph/detail/host/maximal_independent_set.h>
#include <cusp/graph/detail/device/maximal_independent_set.h>

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
template <typename Matrix, typename Array>
size_t maximal_independent_set(const Matrix& A, Array& stencil, size_t k, cusp::host_memory)
{
    return cusp::graph::detail::host::maximal_independent_set(A, stencil, k);
}

//////////////////
// Device Paths //
//////////////////
template <typename Matrix, typename Array>
size_t maximal_independent_set(const Matrix& A, Array& stencil, size_t k, cusp::device_memory)
{
    return cusp::graph::detail::device::maximal_independent_set(A, stencil, k);
}

} // end namespace dispatch
} // end namespace detail
} // end namespace graph
} // end namespace cusp

