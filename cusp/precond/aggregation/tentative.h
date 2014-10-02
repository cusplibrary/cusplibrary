/*
 *  Copyright 2008-2014 NVIDIA Corporation
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
namespace precond
{
namespace aggregation
{

template <typename Array1,
         typename Array2,
         typename MatrixType,
         typename Array3>
void fit_candidates(const Array1& aggregates,
                    const Array2& B,
                    MatrixType& Q_,
                    Array3& R);

} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

#include <cusp/precond/aggregation/detail/tentative.inl>

