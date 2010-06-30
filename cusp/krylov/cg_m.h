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

/*! \addtogroup iterative_solvers Iterative Solvers
 *  \addtogroup krylov_methods Krylov Methods
 *  \ingroup iterative_solvers
 *  \{
 */

/*! \p cg_m 
 * \TODO DOCUMENT
 *
 */
template <class LinearOperator,
          class VectorType1,
          class VectorType2,
          class VectorType3>
void cg_m(LinearOperator& A,
          VectorType1& x,
          VectorType2& b,
          VectorType3& sigma);

/*! \p cg_m 
 * \TODO DOCUMENT
 *
 */
template <class LinearOperator,
          class VectorType1,
          class VectorType2,
          class VectorType3,
          class Monitor>
void cg_m(LinearOperator& A,
        VectorType1& x, VectorType2& b, VectorType3& sigma,
        Monitor& monitor);
/*! \}
 */

} // end namespace krylov
} // end namespace cusp

#include <cusp/krylov/detail/cg_m.inl>

