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

#include <cusp/detail/config.h>
#include <thrust/system/detail/generic/select_system.h>

#include <cusp/exception.h>
#include <cusp/graph/symmetric_rcm.h>

#include <cusp/system/detail/adl/graph/symmetric_rcm.h>
#include <cusp/system/detail/generic/graph/symmetric_rcm.h>

namespace cusp
{
namespace graph
{

template<typename DerivedPolicy, typename MatrixType, typename PermutationType>
void symmetric_rcm(const thrust::execution_policy<DerivedPolicy>& exec,
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

template <typename DerivedPolicy, typename MatrixType, typename PermutationType>
void symmetric_rcm(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                   const MatrixType& G,
                   PermutationType& P)
{
    using cusp::system::detail::generic::symmetric_rcm;

    typename MatrixType::format format;

    if(G.num_rows != G.num_cols)
        throw cusp::invalid_input_exception("matrix must be square");

    symmetric_rcm(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), G, P, format);
}

template<typename MatrixType, typename PermutationType>
void symmetric_rcm(const MatrixType& G, PermutationType& P)
{
    using thrust::system::detail::generic::select_system;

    typedef typename MatrixType::memory_space System1;
    typedef typename PermutationType::memory_space  System2;

    System1 system1;
    System2 system2;

    cusp::graph::symmetric_rcm(select_system(system1,system2), G, P);
}

} // end namespace graph
} // end namespace cusp
