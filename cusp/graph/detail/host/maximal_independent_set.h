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

#include <cusp/array1d.h>

namespace cusp
{
namespace graph
{
namespace detail
{
namespace host
{

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

////////////////
// Host Paths //
////////////////
      
template <typename Matrix, typename Array>
size_t maximal_independent_set(const Matrix& A, Array& stencil, size_t k)
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

} // end namespace host
} // end namespace detail
} // end namespace graph
} // end namespace cusp

