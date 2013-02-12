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
    
#include <cusp/graph/detail/host/breadth_first_search.h>
#include <cusp/graph/detail/device/breadth_first_search.h>

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
template<bool MARK_PREDECESSORS, typename MatrixType, typename ArrayType>
void breadth_first_search(const MatrixType& G, const typename MatrixType::index_type src,
         ArrayType& labels, cusp::host_memory)
{
    return cusp::graph::detail::host::breadth_first_search<MARK_PREDECESSORS>(G, src, labels);
}

//////////////////
// Device Paths //
//////////////////
template<bool MARK_PREDECESSORS, typename MatrixType, typename ArrayType>
void breadth_first_search(const MatrixType& G, const typename MatrixType::index_type src,
         ArrayType& labels, cusp::device_memory)
{
    return cusp::graph::detail::device::breadth_first_search<MARK_PREDECESSORS>(G, src, labels);
}

} // end namespace dispatch
} // end namespace detail
} // end namespace graph
} // end namespace cusp

