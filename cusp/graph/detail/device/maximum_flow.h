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

#include <cusp/array1d.h>

#include <cusp/graph/breadth_first_search.h>

#include <cusp/detail/device/arch.h>
#include <cusp/detail/device/common.h>
#include <cusp/detail/device/utils.h>
#include <cusp/detail/device/texture.h>

#include <thrust/scatter.h>
#include <thrust/iterator/constant_iterator.h>

namespace cusp
{
namespace graph
{
namespace detail
{
namespace device
{
#ifdef CUSP_GRAPH_EXPERIMENTAL
namespace detail
{
template <typename IndexType, typename ValueType, unsigned int VECTORS_PER_BLOCK, unsigned int THREADS_PER_VECTOR>
__launch_bounds__(VECTORS_PER_BLOCK * THREADS_PER_VECTOR,1)
__global__ void
bfs_edge_index_locator_kernel(const IndexType num_rows,
                              const IndexType * Ap,
                              const IndexType * Aj,
                              const ValueType * Ax,
                              const IndexType * bfs_tree,
                              IndexType * positions)
{
    __shared__ volatile IndexType ptrs[VECTORS_PER_BLOCK][2];
    __shared__ volatile IndexType perm_row[VECTORS_PER_BLOCK];

    const IndexType THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

    const IndexType thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const IndexType thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const IndexType vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const IndexType vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const IndexType num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

    for(IndexType row = vector_id; row < num_rows; row += num_vectors)
    {
        if(thread_lane == 0)
            perm_row[vector_lane] = bfs_tree[row];

        if(perm_row[vector_lane] < 0) continue;
        // use two threads to fetch Ap[row] and Ap[row+1]
        // this is considerably faster than the straightforward version
        if(thread_lane < 2)
            ptrs[vector_lane][thread_lane] = Ap[perm_row[vector_lane] + thread_lane];

        const IndexType row_start = ptrs[vector_lane][0];                   //same as: row_start = Ap[row];
        const IndexType row_end   = ptrs[vector_lane][1];                   //same as: row_end   = Ap[row+1];

        // check current row neighbors
        for(IndexType jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR)
            if(Aj[jj] == row) {
                positions[row] = jj;
            }
    }
}

template <unsigned int THREADS_PER_VECTOR, typename Matrix, typename ArrayType>
void __bfs_edge_index_locator(const Matrix&    A,
                              const ArrayType& bfs_tree,
                              ArrayType& positions)
{
    typedef typename Matrix::index_type IndexType;
    typedef typename Matrix::value_type ValueType;

    const size_t THREADS_PER_BLOCK  = 128;
    const size_t VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;

    const size_t MAX_BLOCKS = cusp::detail::device::arch::max_active_blocks(bfs_edge_index_locator_kernel<IndexType, ValueType, VECTORS_PER_BLOCK, THREADS_PER_VECTOR>, THREADS_PER_BLOCK, (size_t) 0);
    const size_t NUM_BLOCKS = std::min<size_t>(MAX_BLOCKS, cusp::detail::device::DIVIDE_INTO(A.num_rows, VECTORS_PER_BLOCK));

    bfs_edge_index_locator_kernel<IndexType, ValueType, VECTORS_PER_BLOCK, THREADS_PER_VECTOR> <<<NUM_BLOCKS, THREADS_PER_BLOCK>>>
    (A.num_rows,
     thrust::raw_pointer_cast(&A.row_offsets[0]),
     thrust::raw_pointer_cast(&A.column_indices[0]),
     thrust::raw_pointer_cast(&A.values[0]),
     thrust::raw_pointer_cast(&bfs_tree[0]),
     thrust::raw_pointer_cast(&positions[0]));
}

template <typename Matrix,
         typename ArrayType>
void bfs_edge_index_locator(const Matrix&    A,
                            const ArrayType& bfs_tree,
                            ArrayType& positions)
{
    typedef typename Matrix::index_type IndexType;

    const IndexType nnz_per_row = A.num_entries / A.num_rows;

    if (nnz_per_row <=  2) {
        __bfs_edge_index_locator<2>(A, bfs_tree, positions);
        return;
    }
    if (nnz_per_row <=  4) {
        __bfs_edge_index_locator<4>(A, bfs_tree, positions);
        return;
    }
    if (nnz_per_row <=  8) {
        __bfs_edge_index_locator<8>(A, bfs_tree, positions);
        return;
    }
    if (nnz_per_row <= 16) {
        __bfs_edge_index_locator<16>(A, bfs_tree, positions);
        return;
    }

    __bfs_edge_index_locator<32>(A, bfs_tree, positions);
}
} // end namespace detail

template<typename MatrixType, typename ArrayType, typename IndexType>
typename MatrixType::value_type
maximum_flow(const MatrixType& G, ArrayType& flow, IndexType src, IndexType sink)
{
    typedef typename MatrixType::value_type ValueType;
    typedef typename MatrixType::memory_space MemorySpace;
    typedef cusp::csr_matrix_view<
    typename MatrixType::row_offsets_array_type,
             typename MatrixType::column_indices_array_type,
             typename MatrixType::values_array_type> MatrixType_view;

    const size_t N = G.num_rows;

    // copy of column indices
    cusp::array1d<IndexType,MemorySpace> column_indices(G.column_indices);
    // create view using new column indices
    MatrixType_view G_view(G.num_rows, G.num_cols, G.num_entries,
                           G.row_offsets, column_indices, G.values);

    // initialize device memory arrays
    cusp::array1d<IndexType,MemorySpace> bfs_tree(N);
    cusp::array1d<IndexType,MemorySpace> positions(N);

    // initialize host memory arrays
    cusp::array1d<IndexType,cusp::host_memory> bfs_tree_h(N);
    cusp::array1d<IndexType,cusp::host_memory> positions_h(N);
    cusp::array1d<IndexType,cusp::host_memory> update_positions(N);

    // copy of capacities to host
    cusp::array1d<ValueType,cusp::host_memory> capacities(G.values);

    ValueType max_flow = 0;

    // Edmonds-Karp algorithm
    while(1)
    {
        // Construct BFS tree starting from the source
        cusp::graph::breadth_first_search<true>(G_view, src, bfs_tree);
        bfs_tree_h = bfs_tree;
        // Break when the sink is not reachable from the source
        if( bfs_tree_h[sink] < 0 ) break;

        // BFS implementation returns the parents but we need to update the actual edges
        // Use a custom spmv-like kernel to find positions of BFS tree entries
        detail::bfs_edge_index_locator(G_view, bfs_tree, positions);
        positions_h = positions;

        int path_length = 0;
        IndexType curr_node = sink;
        ValueType min_capacity = std::numeric_limits<ValueType>::max();
        // Traverse BFS tree starting from sink and ending at source
        while(curr_node != src)
        {
            // track the minimum capacity
            min_capacity = std::min(capacities[positions_h[curr_node]], min_capacity);
            curr_node = bfs_tree_h[curr_node];
            path_length++;
        }

        if( min_capacity == 0 )
        {
            throw cusp::runtime_exception("Max-flow path from source to sink has zero capacity.");
        }

        max_flow += min_capacity;

        int zero_capacity_edges = 0;
        curr_node = sink;
        while(curr_node != src)
        {
            // update the capacities using the minimum capacity
            capacities[positions_h[curr_node]] -= min_capacity;

            // track edges with zero capacity for filtering
            if( capacities[positions_h[curr_node]] == 0 )
            {
                update_positions[zero_capacity_edges++] = positions_h[curr_node];
            }

            curr_node = bfs_tree_h[curr_node];
        }

        thrust::copy(update_positions.begin(), update_positions.begin() + zero_capacity_edges, positions.begin());
        // edges with zero capacity are set to invalid vertex id
        thrust::scatter(thrust::constant_iterator<IndexType>(-1),
                        thrust::constant_iterator<IndexType>(-1) + zero_capacity_edges,
                        positions.begin(),
                        G_view.column_indices.begin());
    }

    // copy updated capacities back to device
    flow = capacities;
    // flow is equal to the difference in the capacities
    cusp::blas::axpby(G.values, flow, flow, 1, -1);

    return max_flow;
}
#else
template<typename MatrixType, typename ArrayType, typename IndexType>
typename MatrixType::value_type
maximum_flow(const MatrixType& G, ArrayType& flow, IndexType src, IndexType sink)
{
  throw cusp::runtime_exception("Maximum flow solver on GPU is experimental, not intended for production or benchmark usage.");
}
#endif

} // end namespace device
} // end namespace detail
} // end namespace graph
} // end namespace cusp

