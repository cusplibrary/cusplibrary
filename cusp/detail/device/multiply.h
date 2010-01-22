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

#include <cusp/coo_matrix.h>
#include <cusp/array2d.h>

#include <cusp/detail/generic/multiply.h>

namespace cusp
{
namespace detail
{
namespace device
{

//////////////////////////////////
// Matrix-Matrix Multiplication //
//////////////////////////////////
template <typename IndexType,
          typename ValueType,
          typename SpaceOrAlloc>
void multiply(const cusp::coo_matrix<IndexType,ValueType,SpaceOrAlloc>& A,
              const cusp::coo_matrix<IndexType,ValueType,SpaceOrAlloc>& B,
                    cusp::coo_matrix<IndexType,ValueType,SpaceOrAlloc>& C)
{
    cusp::detail::generic::multiply(A,B,C);
}
    
template <typename ValueType,
          typename SpaceOrAlloc>
void multiply(const cusp::array2d<ValueType,SpaceOrAlloc>& A,
              const cusp::array2d<ValueType,SpaceOrAlloc>& B,
                    cusp::array2d<ValueType,SpaceOrAlloc>& C)
{
    cusp::detail::generic::multiply(A,B,C);
}

//////////////////////////////////
// Matrix-Vector Multiplication //
//////////////////////////////////
//template <typename IndexType,
//          typename ValueType,
//          typename SpaceOrAlloc,
//          typename MatrixOrVector1,
//          typename MatrixOrVector2>
//void multiply(const cusp::csr_matrix<IndexType,ValueType,SpaceOrAlloc>& A,
//              const MatrixOrVector1& B,
//                    MatrixOrVector2& C);

} // end namespace device
} // end namespace detail
} // end namespace cusp

