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

#include <cusp/blas.h>

namespace cusp
{

struct default_stopping_criteria
{
    float b_norm;
    float tolerance;
    size_t iteration_limit;

    default_stopping_criteria()
        : tolerance(1.0e-6), iteration_limit(500) {}
    
    default_stopping_criteria(float tolerance)
        : tolerance(tolerance), iteration_limit(500) {}
    
    default_stopping_criteria(float tolerance, size_t iteration_limit)
        : tolerance(tolerance), iteration_limit(iteration_limit) {}

    template <typename LinearOperator,
              typename VectorType1,
              typename VectorType2>
    void initialize(const LinearOperator& A,
                    const VectorType1& x,
                    const VectorType2& b)
    {
        b_norm = cusp::blas::nrm2(b);
    }

    template <typename LinearOperator,
              typename VectorType1,
              typename VectorType2,
              typename ScalarType>
    bool has_converged(const LinearOperator& A,
                       const VectorType1& x,
                       const VectorType2& b,
                       const ScalarType residual_norm)
    {
        // if ||b|| is zero then use absolute tolerance
        // otherwise use relative tolerance

        if (b_norm == 0)
            return residual_norm < tolerance;
        else
            return residual_norm < tolerance * b_norm;
    }

    bool has_reached_iteration_limit(size_t iteration_number)
    {
        return iteration_number >= iteration_limit;
    }
};

} // end namespace cusp
