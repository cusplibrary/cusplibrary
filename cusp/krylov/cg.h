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

namespace cusp
{
namespace krylov
{

template <class LinearOperator,
          class VectorType>
void cg(LinearOperator& A,
        VectorType& x,
        VectorType& b);

template <class LinearOperator,
          class VectorType,
          class Monitor>
void cg(LinearOperator& A,
        VectorType& x,
        VectorType& b,
        Monitor& monitor);

template <class LinearOperator,
          class VectorType,
          class Monitor,
          class Preconditioner>
void cg(LinearOperator& A,
        VectorType& x,
        VectorType& b,
        Monitor& monitor,
        Preconditioner& M);

} // end namespace krylov
} // end namespace cusp

#include <cusp/krylov/detail/cg.inl>

