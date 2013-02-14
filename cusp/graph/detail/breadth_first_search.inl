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

#include <cusp/exception.h>
#include <cusp/csr_matrix.h>

#include <cusp/graph/detail/dispatch/breadth_first_search.h>

namespace cusp
{
namespace graph
{
namespace detail
{

template<bool MARK_PREDECESSORS, typename MatrixType, typename ArrayType>
void breadth_first_search(const MatrixType& G, const typename MatrixType::index_type src, ArrayType& labels, cusp::csr_format)
{
    cusp::graph::detail::dispatch::breadth_first_search<MARK_PREDECESSORS>(G, src, labels,
            typename MatrixType::memory_space());
}

//////////////////
// General Path //
//////////////////

template<bool MARK_PREDECESSORS, typename MatrixType, typename ArrayType, typename Format>
void breadth_first_search(const MatrixType& G, const typename MatrixType::index_type src, ArrayType& labels, Format)
{
  typedef typename MatrixType::index_type   IndexType;
  typedef typename MatrixType::value_type   ValueType;
  typedef typename MatrixType::memory_space MemorySpace;

  // convert matrix to CSR format and compute on the host
  cusp::csr_matrix<IndexType,ValueType,MemorySpace> G_csr(G);

  cusp::graph::breadth_first_search(G_csr, src, labels);
}

} // end namespace detail

/////////////////
// Entry Point //
/////////////////

template<bool MARK_PREDECESSORS, typename MatrixType, typename ArrayType>
void breadth_first_search(const MatrixType& G, const typename MatrixType::index_type src, ArrayType& labels)
{
    CUSP_PROFILE_SCOPED();

    if(G.num_rows != G.num_cols)
        throw cusp::invalid_input_exception("matrix must be square");

    cusp::graph::detail::breadth_first_search<MARK_PREDECESSORS>(G, src, labels,
            typename MatrixType::format());
}

} // end namespace graph
} // end namespace cusp

