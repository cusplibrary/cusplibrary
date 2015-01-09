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
namespace eigen
{

/*! \addtogroup iterative_solvers Iterative Solvers
 *  \addtogroup eigensolvers EigenSolvers
 *  \ingroup iterative_solvers
 *  \{
 */

/* \cond */
template <class LinearOperator,
         class Vector>
void lobpcg(LinearOperator& A,
        Vector& x,
        Vector& b);

template <class LinearOperator,
         class Vector,
         class Monitor>
void lobpcg(LinearOperator& A,
        Vector& x,
        Vector& b,
        Monitor& monitor);
/* \endcond */

template <class LinearOperator,
         class Vector,
         class Monitor,
         class Preconditioner>
void lobpcg(LinearOperator& A,
            Vector& x,
            Vector& b,
            Monitor& monitor,
            Preconditioner& M);

/*! \}
 */

} // end namespace eigen
} // end namespace cusp

#include <cusp/eigen/detail/lanczos.inl>
