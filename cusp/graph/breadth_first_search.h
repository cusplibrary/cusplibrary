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

/*! \file breadth_first_search.h
 *  \brief Breadth-first traversal of a graph
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

/*! \p breadth_first_search : Performs a Breadth-first traversal of a graph 
 * starting from a given source vertex.
 *
 * \param A symmetric matrix that represents a graph
 * \param source vertex to begin traversal
 * \param labels of vertices from source in BFS order
 *
 * \tparam Matrix matrix
 * \tparam Array array
 *
 *  \see http://en.wikipedia.org/wiki/Breadth-first_search
 */
template<bool MARK_PREDECESSORS, typename MatrixType, typename ArrayType>
void breadth_first_search(const MatrixType& G, const typename MatrixType::index_type src, ArrayType& labels);

/*! \}
 */


} // end namespace graph
} // end namespace cusp

#include <cusp/graph/detail/breadth_first_search.inl>

