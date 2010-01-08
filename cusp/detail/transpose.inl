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


#include <cusp/coo_matrix.h>

#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace cusp
{
namespace detail
{

template <typename IndexType, typename ValueType, typename SpaceOrAlloc>
void transpose(const cusp::coo_matrix<IndexType,ValueType,SpaceOrAlloc>& A,
                     cusp::coo_matrix<IndexType,ValueType,SpaceOrAlloc>& At)
{
    cusp::coo_matrix<IndexType,ValueType,SpaceOrAlloc> temp(A.num_cols, A.num_rows, A.num_entries);

    cusp::array1d<IndexType,SpaceOrAlloc> permutation(A.num_entries);
    thrust::sequence(permutation.begin(), permutation.end());

    temp.row_indices = A.column_indices;

    thrust::stable_sort_by_key(temp.row_indices.begin(), temp.row_indices.end(), permutation.begin());

    thrust::gather(temp.column_indices.begin(), temp.column_indices.end(),
                   permutation.begin(),
                   A.row_indices.begin());
    
    thrust::gather(temp.values.begin(), temp.values.end(),
                   permutation.begin(),
                   A.values.begin());

    At.swap(temp);
}
 
} // end namespace detail

template <typename MatrixType>
void transpose(const MatrixType& A, MatrixType& At)
{
    cusp::detail::transpose(A, At);
}

} // end namespace cusp

