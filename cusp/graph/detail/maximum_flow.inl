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

#include <cusp/graph/detail/dispatch/maximum_flow.h>

namespace cusp
{
namespace graph
{
namespace detail
{

template<typename MatrixType, typename ArrayType, typename IndexType>
typename MatrixType::value_type 
maximum_flow(const MatrixType& G, ArrayType& flow, const IndexType src, const IndexType sink, cusp::csr_format)
{
    return cusp::graph::detail::dispatch::maximum_flow(G, flow, src, sink,
            typename MatrixType::memory_space());
}

template<typename MatrixType, typename ArrayType1, typename ArrayType2, typename IndexType>
size_t max_flow_to_min_cut(const MatrixType& G, const ArrayType1& flow, const IndexType src, ArrayType2& min_cut_edges, cusp::csr_format)
{
    typedef typename MatrixType::value_type ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    typedef cusp::csr_matrix_view<
    	     typename MatrixType::row_offsets_array_type,
             typename MatrixType::column_indices_array_type,
             typename MatrixType::values_array_type> MatrixType_view;

    cusp::array1d<IndexType,MemorySpace> row_indices(G.num_entries);
    cusp::array1d<ValueType,MemorySpace> residual(G.num_entries);
    cusp::array1d<ValueType,MemorySpace> labels(G.num_rows, -1);

    // construct row indices
    cusp::detail::offsets_to_indices(G.row_offsets, row_indices);

    // copy of column indices
    cusp::array1d<IndexType,MemorySpace> column_indices(G.column_indices);
    // create view using new column indices
    MatrixType_view G_view(G.num_rows, G.num_cols, G.num_entries,
                           G.row_offsets, column_indices, G.values);

    // residual is equal to the difference in capacity and flow
    cusp::blas::axpby(G.values, flow, residual, 1, -1);

    // edges with zero capacity are set to invalid vertex id
    thrust::replace_if(G_view.column_indices.begin(), G_view.column_indices.end(),
    			 residual.begin(), thrust::logical_not<IndexType>(), IndexType(-1));

    // Construct BFS levels starting from the source
    cusp::graph::breadth_first_search<false>(G_view, src, labels);

    // partition vertices into sets marked as -1 or 1
    thrust::replace_if(labels.begin(), labels.end(), thrust::placeholders::_1 != -1, IndexType(1));

    // identify edges spanning both sets 
    thrust::transform(thrust::make_permutation_iterator(labels.begin(), row_indices.begin()),
                      thrust::make_permutation_iterator(labels.begin(), row_indices.end()),
                      thrust::make_permutation_iterator(labels.begin(), G.column_indices.begin()),
                      min_cut_edges.begin(), thrust::not_equal_to<IndexType>());

    return thrust::reduce(min_cut_edges.begin(), min_cut_edges.end());
}

//////////////////
// General Path //
//////////////////

template<typename MatrixType, typename ArrayType, typename IndexType, typename Format>
typename MatrixType::value_type 
size_tmaximum_flow(const MatrixType& G, ArrayType& flow, const IndexType src, const IndexType sink, Format)
{
  typedef typename MatrixType::value_type   ValueType;
  typedef typename MatrixType::memory_space MemorySpace;

  // convert matrix to CSR format and compute on the host
  cusp::csr_matrix<IndexType,ValueType,MemorySpace> G_csr(G);

  return cusp::graph::maximum_flow(G_csr, flow, src, sink);
}

template<typename MatrixType, typename ArrayType1, typename ArrayType2, typename IndexType, typename Format>
size_t max_flow_to_min_cut(const MatrixType& G, const ArrayType1& flow, const IndexType src, ArrayType2& min_cut_edges, Format)
{
  typedef typename MatrixType::value_type   ValueType;
  typedef typename MatrixType::memory_space MemorySpace;

  // convert matrix to CSR format and compute on the host
  cusp::csr_matrix<IndexType,ValueType,MemorySpace> G_csr(G);

  return cusp::graph::max_flow_to_min_cut(G_csr, flow, src, min_cut_edges);
}

} // end namespace detail

/////////////////
// Entry Point //
/////////////////

template<typename MatrixType, typename ArrayType, typename IndexType>
typename MatrixType::value_type
maximum_flow(const MatrixType& G, ArrayType& flow, const IndexType src, const IndexType sink)
{
    CUSP_PROFILE_SCOPED();

    if(G.num_rows != G.num_cols)
        throw cusp::invalid_input_exception("matrix must be square");

    return cusp::graph::detail::maximum_flow(G, flow, src, sink,
            typename MatrixType::format());
}


template<typename MatrixType, typename IndexType>
typename MatrixType::value_type
maximum_flow(const MatrixType& G, const IndexType src, const IndexType sink)
{
    typedef typename MatrixType::value_type ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    cusp::array1d<ValueType,MemorySpace> flow(G.num_entries);

    return cusp::graph::detail::maximum_flow(G, flow, src, sink,
            typename MatrixType::format());
}

template<typename MatrixType, typename ArrayType1, typename ArrayType2, typename IndexType>
size_t max_flow_to_min_cut(const MatrixType& G, const ArrayType1& flow, const IndexType src, ArrayType2& min_cut_edges)
{
    CUSP_PROFILE_SCOPED();

    if(G.num_rows != G.num_cols)
        throw cusp::invalid_input_exception("matrix must be square");

    return cusp::graph::detail::max_flow_to_min_cut(G, flow, src, min_cut_edges,
            typename MatrixType::format());
}

} // end namespace graph
} // end namespace cusp

