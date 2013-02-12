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

/*! \file rcm.h
 *  \brief Reverse Cuthill-Mckee of a sparse matrix
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

/*! \p rcm : Performs a reordering on a graph represented by a symmetric sparse
 * adjacency matrix in order to decrease the bandwidth. The reordering is computed
 * using the Cuthill-McKee algorithm and reversing the resulting index numbers.
 *
 * \param A symmetric matrix that represents a graph
 *
 * \tparam Matrix matrix
 *
 *  \see http://en.wikipedia.org/wiki/Cuthill-McKee_algorithm
 */
template<typename MatrixType>
void symmetric_rcm(MatrixType& G);

/*! \}
 */


} // end namespace graph
} // end namespace cusp

#include <cusp/graph/detail/symmetric_rcm.inl>

