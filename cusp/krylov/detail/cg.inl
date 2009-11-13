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


#include <cusp/array1d.h>
#include <cusp/blas.h>
#include <cusp/spblas.h>
#include <cusp/stopping_criteria.h>

#include <iostream>

namespace blas = cusp::blas;

namespace cusp
{
namespace krylov
{

    
template <class MatrixType,
          class VectorType>
void cg(const MatrixType& A,
              VectorType& x,
        const VectorType& b)
{
    return cg(A, x, b, cusp::default_stopping_criteria());
}


template <class MatrixType,
          class VectorType,
          class StoppingCriteria>
void cg(const MatrixType& A,
              VectorType& x,
        const VectorType& b,
        StoppingCriteria stopping_criteria,
        const int verbose)
{
    typedef typename MatrixType::value_type   ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    assert(A.num_rows == A.num_cols);        // sanity check

    const size_t N = A.num_rows;

    // allocate workspace
    cusp::array1d<ValueType,MemorySpace> y(N);
    cusp::array1d<ValueType,MemorySpace> r(N);
    cusp::array1d<ValueType,MemorySpace> p(N);
        
    //clock_t start = clock();

    // initialize the stopping criteria
    stopping_criteria.initialize(A, x, b);
   
    // y <- Ax
    blas::fill(y, 0);
    cusp::spblas::spmv(A, x, y);

    // r <- b - A*x
    blas::axpby(b, y, r, static_cast<ValueType>(1.0), static_cast<ValueType>(-1.0));
   
    // p <- r
    blas::copy(r, p);
		
    ValueType r_norm = blas::nrm2(r.begin(), r.end());
    ValueType r2     = r_norm * r_norm;                     // <r,r>

    if (verbose)
        std::cout << "[CG] initial residual norm " << r_norm << std::endl;

    size_t iteration_number = 0;

    while (true)
    {
        if (stopping_criteria.has_converged(A, x, b, r_norm))
        {
            if (verbose)
                    std::cout << "[CG] converged in " << iteration_number << " iterations (achieved " << r_norm << " residual)" << std::endl;
            break;
        }
        
        if (stopping_criteria.has_reached_iteration_limit(iteration_number))
        {
            if (verbose)
                    std::cout << "[CG] failed to converge within " << iteration_number << " iterations (achieved " << r_norm << " residual)" << std::endl;;
            break;
        }

        // y <- Ap
        blas::fill(y, 0);
        cusp::spblas::spmv(A, p, y);

        // alpha <- <r,r>/<y,p>
        ValueType alpha =  r2 / blas::dot(y, p);
        // x <- x + alpha * p
        blas::axpy(p, x, alpha);
        // r <- r - alpha * y		
        blas::axpy(y, r, -alpha);
		
        ValueType r2_old = r2;

        // r2 = <r,r>
        r_norm = blas::nrm2(r);
        r2 = r_norm * r_norm;

        // beta <- <r_{i+1},r_{i+1}>/<r,r> 
        ValueType beta = r2 / r2_old;                       
		
        // p <- r + beta*p
        blas::axpby(r, p, p, static_cast<ValueType>(1.0), beta);

        iteration_number++;
    }

    //cudaThreadSynchronize();

    // MFLOPs excludes BLAS operations
    //double elapsed = ((double) (clock() - start)) / CLOCKS_PER_SEC;
    //double MFLOPs = 2* ((double) i * (double) A.num_entries)/ (1e6 * elapsed);
    //printf("-iteration completed in %lfms  ( > %6.2lf MFLOPs )\n",1000*elapsed, MFLOPs );
}

} // end namespace krylov
} // end namespace cusp

