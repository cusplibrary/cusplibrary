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

/*! \file pseudo_peripheral.h
 *  \brief Pseduo peripheral vertex of a graph
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

/*! \p pseudo_peripheral_vertex : finds a pseduo-peripheral vertex
 * a graph. The pseduo-peripheral vertex is the vertex which achieves
 * the diameter of the graph, i.e. achieves the maximum separation distance.
 *
 * \param A symmetric matrix that represents a graph
 * \param BFS level set of vertices starting from pseudo-peripheral vertex
 *
 * \tparam Matrix matrix
 * \tparam Array array
 *
 *  \see http://en.wikipedia.org/wiki/Distance_(graph_theory)
 */
template<typename MatrixType>
typename MatrixType::index_type pseudo_peripheral_vertex(const MatrixType& G);

template<typename MatrixType, typename ArrayType>
typename MatrixType::index_type pseudo_peripheral_vertex(const MatrixType& G, ArrayType& levels);

/*! \}
 */


} // end namespace graph
} // end namespace cusp

#include <cusp/graph/detail/pseudo_peripheral.inl>

