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

/*! \file lobpcg.h
 *  \brief LOBPCG method
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
            Vector& S,
            Vector& X,
            bool largest = true);

template <class LinearOperator,
         class Vector,
         class Monitor>
void lobpcg(LinearOperator& A,
            Vector& S,
            Vector& X,
            Monitor& monitor,
            bool largest = true);
/* \endcond */

/**
 * \brief LOBPCG method
 *
 * \tparam LinearOperator is a matrix or subclass of \p linear_operator
 * \tparam Vector vector
 * \tparam Monitor is a \p monitor
 * \tparam Preconditioner is a matrix or subclass of \p linear_operator
 *
 * \param A matrix of the linear system
 * \param S eigenvalues
 * \param X eigenvectors
 * \param monitor monitors iteration and determines stopping conditions
 * \param M preconditioner for A
 * \param largest If true compute the eigenpair corresponding to the largest
 * eigenvalue otherwise compute the smallest.
 *
 * \par Overview
 * Computes the extreme eigenpairs of hermitian linear systems A x = s x
 * with preconditioner \p M.
 *
 * \note \p A and \p M must be symmetric.
 *
 * \par Example
 *  The following code snippet demonstrates how to use \p lobpcg to
 *  solve a 10x10 Poisson problem.
 *
 *  \code
 *  #include <cusp/csr_matrix.h>
 *  #include <cusp/monitor.h>
 *  #include <cusp/eigen/lobpcg.h>
 *  #include <cusp/gallery/poisson.h>
 *
 *  int main(void)
 *  {
 *      // create an empty sparse matrix structure (CSR format)
 *      cusp::csr_matrix<int, double, cusp::device_memory> A;
 *
 *      // initialize matrix
 *      cusp::gallery::poisson5pt(A, 10, 10);
 *
 *      // allocate storage and initialize eigenpairs
 *      cusp::random_array<double> randx(A.num_rows);
 *      cusp::array1d<double, cusp::device_memory> X(randx);
 *      cusp::array1d<double, cusp::device_memory> S(1,0);
 *
 *      // set stopping criteria:
 *      //  iteration_limit    = 100
 *      //  relative_tolerance = 1e-6
 *      //  absolute_tolerance = 0
 *      //  verbose            = true
 *      cusp::monitor<double> monitor(X, 100, 1e-6, 0, true);
 *
 *      // set preconditioner (identity)
 *      cusp::identity_operator<double, cusp::device_memory> M(A.num_rows, A.num_rows);
 *
 *      // Compute the largest eigenpair of A
 *      cusp::eigen::lobpcg(A, S, X, monitor, M, true);
 *
 *      std::cout << "Largest eigenvalue : " << S[0] << std::endl;
 *
 *      return 0;
 *  }
 *  \endcode
 *
 *  \see \p monitor
 *
 */
template <class LinearOperator,
         class Vector,
         class Monitor,
         class Preconditioner>
void lobpcg(LinearOperator& A,
            Vector& S,
            Vector& X,
            Monitor& monitor,
            Preconditioner& M,
            bool largest = true);

/*! \}
 */

} // end namespace eigen
} // end namespace cusp

#include <cusp/eigen/detail/lobpcg.inl>
