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

/*! \file maximal_independent_set.h
 *  \brief Maximal independent set of a graph
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/array1d.h>
#include <cusp/graph/pseudo_peripheral.h>

#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>

namespace cusp
{
namespace graph
{

template<typename MatrixType>
void rcm(MatrixType& G)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    cusp::coo_matrix<IndexType,ValueType,MemorySpace> G_coo(G);

    cusp::array1d<IndexType,MemorySpace> levels(G.num_rows);
    cusp::graph::detail::pseudo_peripheral_vertex(G, levels);

    cusp::array1d<IndexType,MemorySpace> perm(G.num_rows);
    thrust::sequence(perm.begin(), perm.end());
    thrust::sort_by_key(levels.begin(), levels.end(), perm.begin()); 
 
    thrust::gather(G_coo.row_indices.begin(), G_coo.row_indices.end(), perm.begin(), G_coo.row_indices.begin());
    thrust::gather(G_coo.column_indices.begin(), G_coo.column_indices.end(), perm.begin(), G_coo.column_indices.begin());
    G_coo.sort_by_row_and_column();
    G = G_coo;
}

} // end namespace graph
} // end namespace cusp
