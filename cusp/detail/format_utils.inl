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
    
#include <cusp/csr_matrix.h>

#include <thrust/fill.h>
#include <thrust/binary_search.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>

namespace cusp
{
namespace detail
{

template <typename OffsetArray, typename IndexArray>
void offsets_to_indices(const OffsetArray& offsets, IndexArray& indices)
{
    typedef typename OffsetArray::value_type OffsetType;

    // convert compressed row offsets into uncompressed row indices
    thrust::upper_bound(offsets.begin() + 1,
                        offsets.end(),
                        thrust::counting_iterator<OffsetType>(0),
                        thrust::counting_iterator<OffsetType>(indices.size()),
                        indices.begin());
}

template <typename IndexArray, typename OffsetArray>
void indices_to_offsets(const IndexArray& indices, OffsetArray& offsets)
{
    typedef typename OffsetArray::value_type OffsetType;

    // convert uncompressed row indices into compressed row offsets
    thrust::lower_bound(indices.begin(),
                        indices.end(),
                        thrust::counting_iterator<OffsetType>(0),
                        thrust::counting_iterator<OffsetType>(offsets.size()),
                        offsets.begin());
}


template <typename IndexType, typename ValueType, typename SpaceOrAlloc,
          typename ArrayType>
void extract_diagonal(const cusp::csr_matrix<IndexType,ValueType,SpaceOrAlloc>& A, ArrayType& output)
{
    output.resize(thrust::min(A.num_rows, A.num_cols));

    // initialize output to zero
    thrust::fill(output.begin(), output.end(), ValueType(0));

    // first expand the compressed row offsets into row indices
    cusp::array1d<IndexType,SpaceOrAlloc> row_indices(A.num_entries);
    offsets_to_indices(A.row_offsets, row_indices);

    // determine which matrix entries correspond to the matrix diagonal
    cusp::array1d<unsigned int,SpaceOrAlloc> is_diagonal(A.num_entries);
    thrust::transform(row_indices.begin(), row_indices.end(), A.column_indices.begin(), is_diagonal.begin(), thrust::equal_to<IndexType>());

    // scatter the diagonal values to output
    thrust::scatter_if(A.values.begin(), A.values.end(),
                       row_indices.begin(),
                       is_diagonal.begin(),
                       output.begin());
}

} // end namespace detail
} // end namespace cusp

