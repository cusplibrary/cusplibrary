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

#include <cusp/detail/config.h>

#include <cusp/array1d.h>
#include <cusp/exception.h>
#include <cusp/graph/pseudo_peripheral.h>

#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>

namespace cusp
{
namespace graph
{
namespace detail
{

template<typename MatrixType, typename PermutationType>
void symmetric_rcm(const MatrixType& G, PermutationType& P, cusp::csr_format)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    // find peripheral vertex and return BFS levels from vertex
    cusp::graph::pseudo_peripheral_vertex(G, P.permutation);

    // sort vertices by level in BFS traversal
    cusp::array1d<IndexType,MemorySpace> levels(G.num_rows);
    thrust::sequence(levels.begin(), levels.end());
    thrust::sort_by_key(P.permutation.begin(), P.permutation.end(), levels.begin());
    // form RCM permutation matrix
    thrust::scatter(thrust::counting_iterator<IndexType>(0),
                    thrust::counting_iterator<IndexType>(G.num_rows),
                    levels.begin(), P.permutation.begin());
}

//////////////////
// General Path //
//////////////////

template<typename MatrixType, typename PermutationType, typename Format>
void symmetric_rcm(const MatrixType& G, PermutationType& P, Format& format)
{
    typedef typename MatrixType::index_type   IndexType;
    typedef typename MatrixType::value_type   ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    // convert matrix to CSR format and compute on the host
    cusp::csr_matrix<IndexType,ValueType,MemorySpace> G_csr(G);

    cusp::graph::symmetric_rcm(G_csr, P);
}

} // end namespace detail

/////////////////
// Entry Point //
/////////////////

template<typename MatrixType, typename PermutationType>
void symmetric_rcm(const MatrixType& G, PermutationType& P)
{
    CUSP_PROFILE_SCOPED();

    if(G.num_rows != G.num_cols)
        throw cusp::invalid_input_exception("matrix must be square");

    cusp::graph::detail::symmetric_rcm(G, P, typename MatrixType::format());
}

} // end namespace graph
} // end namespace cusp
