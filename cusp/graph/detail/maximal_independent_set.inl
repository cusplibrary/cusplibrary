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

#include <cusp/detail/device/generalized_spmv/coo_flat.h>

#include <cusp/array1d.h>
#include <cusp/exception.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>

#include <cusp/print.h>

#include <thrust/count.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

namespace cusp
{
namespace graph
{
namespace detail
{

// http://burtleburtle.net/bob/hash/integer.html
struct simple_hash
{
    __host__ __device__
    unsigned int operator()(unsigned int a)
    {
        a = (a + 0x7ed55d16) + (a << 12);
        a = (a ^ 0xc761c23c) ^ (a >> 19);
        a = (a + 0x165667b1) + (a <<  5);
        a = (a + 0xd3a2646c) ^ (a <<  9);
        a = (a + 0xfd7046c5) + (a <<  3);
        a = (a ^ 0xb55a4f09) ^ (a >> 16);
        return a;
    }
};


struct process_mis_nodes
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        if (thrust::get<1>(t) == 1)                     // undecided node
        {
            if (thrust::get<0>(t) == thrust::get<3>(t)) // i == maximal_index
              thrust::get<1>(t) = 2;                    // mis_node
        }
    }
};

struct process_non_mis_nodes
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        if (thrust::get<0>(t) == 1)            // undecided node
        {
            if (thrust::get<1>(t) == 2)        // maximal_state == mis_node
              thrust::get<0>(t) = 0;           // non_mis_node
        }
    }
};

  
template <typename Matrix, typename IndexType>
void propagate_distances(const Matrix& A,
                         const IndexType i,
                         const size_t d,
                         const size_t k,
                         cusp::array1d<size_t,cusp::host_memory>& distance)
{
  distance[i] = d;

  if (d < k)
  {
    for(IndexType jj = A.row_offsets[i]; jj < A.row_offsets[i + 1]; jj++)
    {
      IndexType j = A.column_indices[jj];

      // update only if necessary
      if (d + 1 < distance[j])
        propagate_distances(A, j, d + 1, k, distance);
    }
  }
}

template <typename NodeStateType>
struct is_active_edge
{
  template <typename Tuple>
  __host__ __device__
  bool operator()(const Tuple& t) const
  {
    return thrust::get<0>(t) == 1 && thrust::get<1>(t) == 1;
  }
};

template <typename NodeStateType>
struct is_active_node
{
  __host__ __device__
  bool operator()(const NodeStateType& s) const
  {
    return s == 1;
  }
};
 

////////////////
// Host Paths //
////////////////
      
template <typename Matrix, typename ArrayType>
size_t maximal_independent_set(const Matrix& A, ArrayType& stencil, size_t k,
                               cusp::csr_format, cusp::host_memory)
{
  typedef typename Matrix::index_type   IndexType;
  
  const IndexType N = A.num_rows;

  // distance to nearest MIS node
  cusp::array1d<size_t,cusp::host_memory> distance(N, k + 1);

  // count number of MIS nodes
  size_t set_nodes = 0;
  
  // pick MIS-k nodes greedily and deactivate all their k-neighbors
  for(IndexType i = 0; i < N; i++)
  {
    if (distance[i] > k)
    {
      set_nodes++;

      // reset distances on all k-ring neighbors 
      propagate_distances(A, i, 0, k, distance);
    }
  }
  
  // write output
  stencil.resize(N);

  for (IndexType i = 0; i < N; i++)
      stencil[i] = distance[i] == 0;

  return set_nodes;
}

//////////////////
// Device Paths //
//////////////////

template <typename Matrix, typename ArrayType>
size_t maximal_independent_set(const Matrix& A, ArrayType& stencil, size_t k,
                               cusp::coo_format, cusp::device_memory)
{
    typedef typename Matrix::index_type   IndexType;
    typedef typename Matrix::value_type   ValueType;
    typedef typename Matrix::memory_space MemorySpace;
    typedef unsigned int RandomType;
    typedef unsigned int NodeStateType;
        
    const IndexType N = A.num_rows;
    
    cusp::array1d<RandomType,MemorySpace> random_values(N);
    thrust::transform(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(N), 
                      random_values.begin(),
                      simple_hash());

    cusp::array1d<NodeStateType,MemorySpace> states(N, 1);
    
    typedef typename thrust::tuple<NodeStateType,RandomType,IndexType> Tuple;
    
    cusp::array1d<NodeStateType,MemorySpace> maximal_states(N);
    cusp::array1d<RandomType,MemorySpace>    maximal_values(N);
    cusp::array1d<IndexType,MemorySpace>     maximal_indices(N);

    cusp::array1d<NodeStateType,MemorySpace> last_states(maximal_states);
    cusp::array1d<RandomType,MemorySpace>    last_values(maximal_values);
    cusp::array1d<IndexType,MemorySpace>     last_indices(maximal_indices);

    // TODO choose threshold in a more principled manner
    size_t compaction_threshold = 0; // (N < 10000) ? 0 : (N / 10);
    size_t active_nodes = N;

    do
    {
        // find the largest (state,value,index) 1-ring neighbor for each node
        cusp::detail::device::cuda::spmv_coo
            (A.num_rows, A.num_entries,
             A.row_indices.begin(), A.column_indices.begin(), thrust::constant_iterator<int>(1),  // XXX should we mask explicit zeros? (e.g. DIA, array2d)
             thrust::make_zip_iterator(thrust::make_tuple(states.begin(), random_values.begin(), thrust::counting_iterator<IndexType>(0))),
             thrust::make_zip_iterator(thrust::make_tuple(states.begin(), random_values.begin(), thrust::counting_iterator<IndexType>(0))),
             thrust::make_zip_iterator(thrust::make_tuple(maximal_states.begin(), maximal_values.begin(), maximal_indices.begin())),
             thrust::project2nd<Tuple,Tuple>(), thrust::maximum<Tuple>());

        // find the largest (state,value,index) k-ring neighbor for each node (if k > 1)
        for(size_t ring = 1; ring < k; ring++)
        {
            last_states.resize (N); last_states.swap (maximal_states);
            last_values.resize (N); last_values.swap (maximal_values);
            last_indices.resize(N); last_indices.swap(maximal_indices);

            // TODO replace with call to generalized method
            cusp::detail::device::cuda::spmv_coo
                (A.num_rows, A.num_entries,
                 A.row_indices.begin(), A.column_indices.begin(), thrust::constant_iterator<int>(1),  // XXX should we mask explicit zeros? (e.g. DIA, array2d)
                 thrust::make_zip_iterator(thrust::make_tuple(last_states.begin(), last_values.begin(), last_indices.begin())),
                 thrust::make_zip_iterator(thrust::make_tuple(last_states.begin(), last_values.begin(), last_indices.begin())),
                 thrust::make_zip_iterator(thrust::make_tuple(maximal_states.begin(), maximal_values.begin(), maximal_indices.begin())),
                 thrust::project2nd<Tuple,Tuple>(), thrust::maximum<Tuple>());
        }
       
        // label local maxima as MIS nodes
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<IndexType>(0), states.begin(), maximal_states.begin(), maximal_indices.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<IndexType>(0), states.begin(), maximal_states.begin(), maximal_indices.begin())) + N,
                         process_mis_nodes());
        
        // label k-ring neighbors of MIS nodes as non-MIS nodes
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(states.begin(), thrust::make_permutation_iterator(states.begin(), maximal_indices.begin()))),
                         thrust::make_zip_iterator(thrust::make_tuple(states.begin(), thrust::make_permutation_iterator(states.begin(), maximal_indices.begin()))) + N,
                         process_non_mis_nodes());

        active_nodes = thrust::count(states.begin(), states.end(), 1);
    } while (active_nodes > compaction_threshold);

    // resize output
    stencil.resize(N);

    // mark all mis nodes
    thrust::transform(states.begin(), states.end(), thrust::constant_iterator<NodeStateType>(2), stencil.begin(), thrust::equal_to<NodeStateType>());
    
    // return the size of the MIS
    return thrust::count(stencil.begin(), stencil.end(), typename ArrayType::value_type(true));
}


//////////////////
// General Path //
//////////////////

template <typename Matrix, typename ArrayType,
          typename Format, typename MemorySpace>
size_t maximal_independent_set(const Matrix& A, ArrayType& stencil, size_t k,
                               Format, MemorySpace)
{
  typedef typename Matrix::index_type   IndexType;
  typedef typename Matrix::value_type   ValueType;

  // convert matrix to CSR format and compute on the host
  cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> A_csr(A);

  return cusp::graph::maximal_independent_set(A_csr, stencil, k);
}

} // end namespace detail

/////////////////
// Entry Point //
/////////////////

template <typename Matrix, typename ArrayType>
size_t maximal_independent_set(const Matrix& A, ArrayType& stencil, size_t k)
{
    CUSP_PROFILE_SCOPED();

    if(A.num_rows != A.num_cols)
        throw cusp::invalid_input_exception("matrix must be square");

    if (k == 0)
    {
        stencil.resize(A.num_rows);
        thrust::fill(stencil.begin(), stencil.end(), typename ArrayType::value_type(1));
        return stencil.size();
    }
    else
    {
        return cusp::graph::detail::maximal_independent_set(A, stencil, k, typename Matrix::format(), typename Matrix::memory_space());
    }
}

} // end namespace graph
} // end namespace cusp

