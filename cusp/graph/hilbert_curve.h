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

/*! \file hilbert_curve.h
 *  \brief Cluster points using a Hilbert space filling curve
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

/*! \p hilbert_curve : Uses a Hilbert space filling curve to partition
 * a set of points in 2 or 3 dimensional space.
 *
 *
 * \param Set of points in 2 or 3-D space
 * \param number of partitions to construct
 * \param partition assigned to each point
 *
 * \tparam Array coord
 * \tparam size_t num_parts
 * \tparam Array parts
 *
 *  \see http://en.wikipedia.org/wiki/Hilbert_curve
 */
template <class Array2d, class Array1d>
void hilbert_curve(const Array2d& coord, const size_t num_parts, Array1d& parts);

/*! \}
 */


} // end namespace graph
} // end namespace cusp

#include <cusp/graph/detail/hilbert_curve.inl>

