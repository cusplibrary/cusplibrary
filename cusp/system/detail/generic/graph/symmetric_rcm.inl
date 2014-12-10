/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

#include <cusp/detail/config.h>

#include <cusp/graph/pseudo_peripheral.h>

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

template<typename DerivedPolicy, typename MatrixType, typename PermutationType>
void symmetric_rcm(thrust::execution_policy<DerivedPolicy>& exec,
                   const MatrixType& G,
                   PermutationType& P,
                   cusp::csr_format)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    // find peripheral vertex and return BFS levels from vertex
    cusp::graph::pseudo_peripheral_vertex(exec, G, P.permutation);

    // sort vertices by level in BFS traversal
    cusp::array1d<IndexType,MemorySpace> levels(G.num_rows);
    thrust::sequence(exec, levels.begin(), levels.end());
    thrust::sort_by_key(exec, P.permutation.begin(), P.permutation.end(), levels.begin());

    // form RCM permutation matrix
    thrust::scatter(exec,
                    thrust::counting_iterator<IndexType>(0),
                    thrust::counting_iterator<IndexType>(G.num_rows),
                    levels.begin(), P.permutation.begin());
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp
