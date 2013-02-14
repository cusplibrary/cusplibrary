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

/*! \file connected_components.h
 *  \brief Compute the connected components of a graph
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

/*! \p connected_components : Computes the connected components of a graph 
 *
 * \param A symmetric matrix that represents a graph
 * \param component each vertex is connected to
 *
 * \tparam Matrix matrix
 * \tparam Array array
 *
 *  \see http://en.wikipedia.org/wiki/Connected_component_(graph_theory)
 */
template<typename MatrixType, typename ArrayType>
size_t connected_components(const MatrixType& G, ArrayType& components);

/*! \}
 */


} // end namespace graph
} // end namespace cusp

#include <cusp/graph/detail/connected_components.inl>

