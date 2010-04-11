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


#pragma once

#include <cusp/detail/config.h>

#include <cusp/gallery/stencil.h>

namespace cusp
{
namespace gallery
{
/*! \addtogroup gallery Matrix Gallery
 *  \addtogroup poisson Poisson
 *  \ingroup gallery
 *  \{
 */

/*! \p poisson5pt: Create a matrix representing a Poisson problem
 * discretized on an \p m by \p n grid with the standard 5-point
 * finite-difference stencil.
 *
 * \param matrix output
 * \param m number of grid rows
 * \param n number of grid columns 
 * \tparam MatrixType matrix container
 *
 * \code
 * #include <cusp/gallery/poisson.h>
 * #include <cusp/coo_matrix.h>
 * #include <cusp/print.h>
 * 
 * int main(void)
 * {
 *     cusp::coo_matrix<int, float, cusp::device_memory> A;
 *     
 *     // create a matrix for a Poisson problem on a 4x4 grid
 *     cusp::gallery::poisson5pt(A, 4, 4);
 * 
 *     // print matrix
 *     cusp::print_matrix(A);
 * 
 *     return 0;
 * }
 * \endcode
 *
 */
template <typename MatrixType>
void poisson5pt(      MatrixType& matrix, size_t m, size_t n)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType; 
    typedef thrust::tuple<IndexType,IndexType>    StencilIndex;
    typedef thrust::tuple<StencilIndex,ValueType> StencilPoint;

    cusp::array1d<StencilPoint, cusp::host_memory> stencil;
    stencil.push_back(StencilPoint(StencilIndex(  0, -1), -1));
    stencil.push_back(StencilPoint(StencilIndex( -1,  0), -1));
    stencil.push_back(StencilPoint(StencilIndex(  0,  0),  4));
    stencil.push_back(StencilPoint(StencilIndex(  1,  0), -1));
    stencil.push_back(StencilPoint(StencilIndex(  0,  1), -1));

    cusp::gallery::generate_matrix_from_stencil(matrix, stencil, StencilIndex(m,n));
}
/*! \}
 */

} // end namespace gallery
} // end namespace cusp

