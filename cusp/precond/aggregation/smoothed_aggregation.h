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
#include <cusp/csr_matrix.h>
#include <cusp/hyb_matrix.h>

#include <cusp/relaxation/jacobi.h>
#include <cusp/relaxation/polynomial.h>

#include <cusp/detail/lu.h>
#include <cusp/detail/spectral_radius.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{

/*! \addtogroup preconditioners Preconditioners
 *  \ingroup preconditioners
 *  \{
 */

template <typename IndexType, typename ValueType, typename MemorySpace>
struct amg_container{};

template <typename IndexType, typename ValueType>
struct amg_container<IndexType,ValueType,cusp::host_memory>
{
    // use CSR on host
    typedef typename cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> setup_type;
    typedef typename cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> solve_type;
};

template <typename IndexType, typename ValueType>
struct amg_container<IndexType,ValueType,cusp::device_memory>
{
    // use COO on device
    typedef typename cusp::coo_matrix<IndexType,ValueType,cusp::device_memory> setup_type;
    typedef typename cusp::hyb_matrix<IndexType,ValueType,cusp::device_memory> solve_type;
};

template <typename SmootherType>
struct SmootherInitializer {};

template <typename ValueType, typename MemorySpace>
struct SmootherInitializer< cusp::relaxation::jacobi<ValueType,MemorySpace> >
{
    typedef cusp::relaxation::jacobi<ValueType,MemorySpace> Smoother;

    template <typename MatrixType>
    Smoother operator()(const MatrixType& A, const ValueType rho_DinvA)
    {
        //  4/3 * 1/rho is a good default, where rho is the spectral radius of D^-1(A)
    	ValueType omega = ValueType(4.0/3.0) / rho_DinvA;
        return Smoother(A, omega);
    }
};

template <typename ValueType, typename MemorySpace>
struct SmootherInitializer< cusp::relaxation::polynomial<ValueType,MemorySpace> >
{
    typedef cusp::relaxation::polynomial<ValueType,MemorySpace> Smoother;

    template <typename MatrixType>
    Smoother operator()(const MatrixType& A, const ValueType rho_DinvA)
    {
        cusp::array1d<ValueType,cusp::host_memory> coef;
        ValueType rho = cusp::detail::ritz_spectral_radius_symmetric(A, 8);
        cusp::relaxation::detail::chebyshev_polynomial_coefficients(rho, coef);
        return Smoother(A, coef);
    }
};


/*! \p smoothed_aggregation : algebraic multigrid preconditoner based on
 *  smoothed aggregation
 *
 *  TODO
 */
template <typename IndexType, typename ValueType, typename MemorySpace, typename SmootherType = cusp::relaxation::jacobi<ValueType,MemorySpace> >
class smoothed_aggregation : public cusp::linear_operator<ValueType, MemorySpace, IndexType>
{

    typedef typename amg_container<IndexType,ValueType,MemorySpace>::setup_type SetupMatrixType;
    typedef typename amg_container<IndexType,ValueType,MemorySpace>::solve_type SolveMatrixType;

    struct level
    {
        SetupMatrixType A_; // matrix
        SolveMatrixType R;  // restriction operator
        SolveMatrixType A;  // matrix
        SolveMatrixType P;  // prolongation operator
        cusp::array1d<IndexType,MemorySpace> aggregates;      // aggregates
        cusp::array1d<ValueType,MemorySpace> B;               // near-nullspace candidates
        cusp::array1d<ValueType,MemorySpace> x;               // per-level solution
        cusp::array1d<ValueType,MemorySpace> b;               // per-level rhs
        cusp::array1d<ValueType,MemorySpace> residual;        // per-level residual
        
        SmootherType smoother;
    };

    SmootherInitializer<SmootherType> smoother_initializer;

    cusp::detail::lu_solver<ValueType, cusp::host_memory> LU;

    ValueType theta;

    public:

    std::vector<level> levels;        

    template <typename MatrixType>
    smoothed_aggregation(const MatrixType& A, const ValueType theta=0);

    template <typename MatrixType, typename ArrayType>
    smoothed_aggregation(const MatrixType& A, const ArrayType& B, const ValueType theta=0);
    
    template <typename Array1, typename Array2>
    void operator()(const Array1& x, Array2& y);

    template <typename Array1, typename Array2>
    void solve(const Array1& b, Array2& x);

    template <typename Array1, typename Array2, typename Monitor>
    void solve(const Array1& b, Array2& x, Monitor& monitor);

    void print( void );

    double operator_complexity( void );

    double grid_complexity( void );

    protected:

    template <typename MatrixType, typename ArrayType>
    void init(const MatrixType& A, const ArrayType& B);

    void extend_hierarchy(void);

    template <typename Array1, typename Array2>
    void _solve(const Array1& b, Array2& x, const size_t i);
};
/*! \}
 */

} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

#include <cusp/precond/aggregation/detail/smoothed_aggregation.inl>

