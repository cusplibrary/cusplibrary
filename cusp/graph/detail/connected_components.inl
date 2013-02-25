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

#include <cusp/graph/detail/dispatch/connected_components.h>

namespace cusp
{
namespace graph
{
namespace detail
{

template<typename MatrixType, typename ArrayType>
size_t connected_components(const MatrixType& G, ArrayType& components, cusp::csr_format)
{
    return cusp::graph::detail::dispatch::connected_components(G, components,
            typename MatrixType::memory_space());
}

//////////////////
// General Path //
//////////////////

template<typename MatrixType, typename ArrayType, typename Format>
size_t connected_components(const MatrixType& G, ArrayType& components, Format)
{
  typedef typename MatrixType::index_type   IndexType;
  typedef typename MatrixType::value_type   ValueType;
  typedef typename MatrixType::memory_space MemorySpace;

  // convert matrix to CSR format and compute on the host
  cusp::csr_matrix<IndexType,ValueType,MemorySpace> G_csr(G);

  return cusp::graph::connected_components(G_csr, components);
}

} // end namespace detail

/////////////////
// Entry Point //
/////////////////

template<typename MatrixType, typename ArrayType>
size_t connected_components(const MatrixType& G, ArrayType& components)
{
    CUSP_PROFILE_SCOPED();

    if(G.num_rows != G.num_cols)
        throw cusp::invalid_input_exception("matrix must be square");

    return cusp::graph::detail::connected_components(G, components,
					 	typename MatrixType::format());
}

} // end namespace graph
} // end namespace cusp

