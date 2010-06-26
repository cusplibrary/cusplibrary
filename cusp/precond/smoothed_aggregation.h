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

/*! \file smoothed_aggregation.h
 *  \brief Algebraic multigrid preconditoner based on smoothed aggregation.
 *  
 */

#pragma once

#include <cusp/detail/config.h>

#include <vector> // TODO replace with host_vector
#include <cusp/linear_operator.h>

#include <cusp/coo_matrix.h>
#include <cusp/relaxation/jacobi.h>

#include <cusp/detail/lu.h>

namespace cusp
{
namespace precond
{

/*! \addtogroup preconditioners Preconditioners
 *  \ingroup preconditioners
 *  \{
 */

/*! \p smoothed_aggregation : algebraic multigrid preconditoner based on
 *  smoothed aggregation
 *
 *  TODO
 */

template <typename IndexType, typename ValueType, typename MemorySpace>
class smoothed_aggregation : public cusp::linear_operator<ValueType, MemorySpace, IndexType>
{
    struct level
    {
        cusp::coo_matrix<IndexType,ValueType,MemorySpace> R;  // restriction operator
        cusp::coo_matrix<IndexType,ValueType,MemorySpace> A;  // matrix
        cusp::coo_matrix<IndexType,ValueType,MemorySpace> P;  // prolongation operator
        cusp::array1d<IndexType,MemorySpace> aggregates;      // aggregates
        cusp::array1d<ValueType,MemorySpace> B;               // near-nullspace candidates
        
        cusp::relaxation::jacobi<ValueType,MemorySpace> smoother;
       
        ValueType rho;                                        // spectral radius
    };

    std::vector<level> levels;
        
    cusp::detail::lu_solver<ValueType, cusp::host_memory> LU;

    public:

    smoothed_aggregation(const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& A);

    
    template <typename Array1, typename Array2>
    void operator()(const Array1& x, Array2& y) const;

    void solve(const cusp::array1d<ValueType,cusp::device_memory>& b,
                     cusp::array1d<ValueType,cusp::device_memory>& x) const;

    protected:

    void extend_hierarchy(void);

    void _solve(const cusp::array1d<ValueType,MemorySpace>& b,
                      cusp::array1d<ValueType,MemorySpace>& x,
                const int i) const;
};
/*! \}
 */

} // end namespace precond
} // end namespace cusp

#include <cusp/precond/detail/smoothed_aggregation.inl>

