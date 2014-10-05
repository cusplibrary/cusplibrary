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

/*! \file monitor.h
 *  \brief Monitor iterative solver convergence
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/blas/blas.h>
#include <cusp/complex.h>

#include <limits>
#include <iostream>
#include <iomanip>

// Classes to monitor iterative solver progress, check for convergence, etc.
// Follows the implementation of Iteration in the ITL:
//   http://www.osl.iu.edu/research/itl/doc/Iteration.html

namespace cusp
{
/*! \addtogroup iterative_solvers Iterative Solvers
 *  \addtogroup monitors Monitors
 *  \ingroup iterative_solvers
 *  \{
 */

/*! \p default_monitor : Implements standard convergence criteria
 * and reporting for iterative solvers.
 *
 * \tparam ValueType scalar type used in the solver (e.g. \c float or \c cusp::complex<double>).
 *
 *  The following code snippet demonstrates how to configure
 *  the \p default_monitor and use it with an iterative solver.
 *
 *  \code
 *  #include <cusp/csr_matrix.h>
 *  #include <cusp/monitor.h>
 *  #include <cusp/krylov/cg.h>
 *  #include <cusp/gallery/poisson.h>
 *
 *  int main(void)
 *  {
 *      // create an empty sparse matrix structure (CSR format)
 *      cusp::csr_matrix<int, float, cusp::device_memory> A;
 *
 *      // initialize matrix
 *      cusp::gallery::poisson5pt(A, 10, 10);
 *
 *      // allocate storage for solution (x) and right hand side (b)
 *      cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0);
 *      cusp::array1d<float, cusp::device_memory> b(A.num_rows, 1);
 *
 *      // set stopping criteria:
 *      //  iteration_limit    = 100
 *      //  relative_tolerance = 1e-6
 *      cusp::monitor<float> monitor(b, 100, 1e-6);
 *
 *      // solve the linear system A x = b
 *      cusp::krylov::cg(A, x, b, monitor);
 *
 *      // report solver results
 *      if (monitor.converged())
 *      {
 *          std::cout << "Solver converged to " << monitor.relative_tolerance() << " relative tolerance";
 *          std::cout << " after " << monitor.iteration_count() << " iterations" << std::endl;
 *      }
 *      else
 *      {
 *          std::cout << "Solver reached iteration limit " << monitor.iteration_limit() << " before converging";
 *          std::cout << " to " << monitor.relative_tolerance() << " relative tolerance " << std::endl;
 *      }
 *
 *      return 0;
 *  }
 *  \endcode
 *
 *  \see \p verbose_monitor
 *
 */
template <typename ValueType>
class monitor
{
public:
    typedef typename cusp::norm_type<ValueType>::type Real;

    /*! Construct a \p default_monitor for a given right-hand-side \p b
     *
     *  The \p default_monitor terminates iteration when the residual norm
     *  satisfies the condition
     *       ||b - A x|| <= absolute_tolerance + relative_tolerance * ||b||
     *  or when the iteration limit is reached.
     *
     *  \param b right-hand-side of the linear system A x = b
     *  \param iteration_limit maximum number of solver iterations to allow
     *  \param relative_tolerance determines convergence criteria
     *  \param absolute_tolerance determines convergence criteria
     *
     *  \tparam VectorType vector
     */
    template <typename Vector>
    monitor(const Vector& b, size_t iteration_limit = 500, Real relative_tolerance = 1e-5, Real absolute_tolerance = 0, bool verbose = false)
        : b_norm(cusp::blas::nrm2(b)),
          r_norm(std::numeric_limits<Real>::max()),
          iteration_limit_(iteration_limit),
          iteration_count_(0),
          relative_tolerance_(relative_tolerance),
          absolute_tolerance_(absolute_tolerance),
          verbose(verbose)
    {
          if(verbose) {
              std::cout << "Solver will continue until ";
              std::cout << "residual norm " << relative_tolerance << " or reaching ";
              std::cout << iteration_limit << " iterations " << std::endl;
              std::cout << "  Iteration Number  | Residual Norm" << std::endl;
          }
          residuals.reserve(iteration_limit);
    }

    /*! increment the iteration count
     */
    void operator++(void) {
        ++iteration_count_;    // prefix increment
    }

    /*! whether the last tested residual satifies the convergence tolerance
     */
    bool converged() const
    {
        return residual_norm() <= tolerance();
    }

    /*! Euclidean norm of last residual
     */
    Real residual_norm() const {
        return r_norm;
    }

    /*! number of iterations
     */
    size_t iteration_count() const {
        return iteration_count_;
    }

    /*! maximum number of iterations
     */
    size_t iteration_limit() const {
        return iteration_limit_;
    }

    /*! relative tolerance
     */
    Real relative_tolerance() const {
        return relative_tolerance_;
    }

    /*! absolute tolerance
     */
    Real absolute_tolerance() const {
        return absolute_tolerance_;
    }

    /*! tolerance
     *
     *  Equal to absolute_tolerance() + relative_tolerance() * ||b||
     *
     */
    Real tolerance() const {
        return absolute_tolerance() + relative_tolerance() * b_norm;
    }

    /*! applies convergence criteria to determine whether iteration is finished
     *
     *  \param r residual vector of the linear system (r = b - A x)
     *  \tparam Vector vector
     */
    template <typename Vector>
    bool finished(const Vector& r);

    void set_verbose(bool verbose_ = true) {
      verbose = verbose_;
    }

    template <typename Vector>
    void reset(const Vector& b) {
        b_norm = cusp::blas::nrm2(b);
        r_norm = std::numeric_limits<Real>::max();
        iteration_count_ = 0;
        residuals.resize(0);
    }

    void print(void);

    Real immediate_rate(void);

    Real geometric_rate(void);

    Real average_rate(void);

protected:

    bool verbose;
    Real r_norm;
    Real b_norm;
    Real relative_tolerance_;
    Real absolute_tolerance_;

    size_t iteration_limit_;
    size_t iteration_count_;
    cusp::array1d<Real,cusp::host_memory> residuals;
};

} // end namespace cusp

#include <cusp/detail/monitor.inl>
