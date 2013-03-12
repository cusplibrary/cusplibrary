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

/*! \file maximum_flow.h
 *  \brief Max-flow in a flow network
 */

#pragma once

#include <cusp/detail/config.h>

namespace cusp
{
namespace graph
{
/*! \addtogroup algorithms Algorithms
 *  \ingroup algorithms
 *  \{
 */

/*! \p maximum_flow : Performs a maximum flow computation on a flow network
 * starting from a given source vertex to a given sink vertex.
 *
 * \param A matrix that represents a flow network
 * \param source vertex
 * \param sink vertex
 *
 * \tparam Matrix matrix
 *
 *  \see http://en.wikipedia.org/wiki/Maximum_flow_problem
 */
template<typename MatrixType, typename IndexType>
typename MatrixType::value_type
maximum_flow(const MatrixType& G, const IndexType src, const IndexType sink);

template<typename MatrixType, typename ArrayType, typename IndexType>
typename MatrixType::value_type
maximum_flow(const MatrixType& G, ArrayType& flow, const IndexType src, const IndexType sink);

template<typename MatrixType, typename ArrayType1, typename ArrayType2, typename IndexType>
size_t max_flow_to_min_cut(const MatrixType& G, const ArrayType1& flow, const IndexType src, ArrayType2& min_cut_edges);

/*! \}
 */


} // end namespace graph
} // end namespace cusp

#include <cusp/graph/detail/maximum_flow.inl>

