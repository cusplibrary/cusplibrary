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

#include <cusp/array1d.h>
#include <cusp/blas.h>

#include <iostream>

namespace blas = cusp::blas;

namespace cusp
{
namespace krylov
{

// TODO use ||b - A * x|| < tol * ||b|| stopping criterion

template <class LinearOperator,
          class Vector>
void cg(LinearOperator A,
              Vector& x,
        const Vector& b,
        const float tol = 1e-5,
        const size_t max_iter = 1000,
        const int verbose  = 0)
{
    typedef typename LinearOperator::value_type   ValueType;
    typedef typename LinearOperator::memory_space MemorySpace;

    assert(A.num_rows == A.num_cols);        // sanity check

    const size_t N = A.num_rows;

    // allocate workspace
    cusp::array1d<ValueType,MemorySpace> y(N);
    cusp::array1d<ValueType,MemorySpace> r(N);
    cusp::array1d<ValueType,MemorySpace> p(N);
        
    //clock_t start = clock();
    
    // y <- Ax
    blas::fill(y.begin(), y.end(), 0);
    A(thrust::raw_pointer_cast(&x[0]), thrust::raw_pointer_cast(&y[0]));                                                 

    // r <- b - A*x
    blas::copy(b.begin(), b.end(), r.begin());
    blas::axpy(y.begin(), y.end(), r.begin(), static_cast<ValueType>(-1.0));
   
    // p <- r
    blas::copy(r.begin(), r.end(), p.begin());
		
    ValueType r_norm = blas::nrm2(r.begin(), r.end());
    ValueType r2     = r_norm * r_norm;                     // <r,r>

    if (verbose)
        std::cout << "[CG] initial residual norm " << r_norm << std::endl;

    ValueType stop_tol = r2*tol*tol;

    size_t i = 0;
    while( i++ < max_iter && r2 > stop_tol )
    {
        // y <- Ap
        blas::fill(y.begin(), y.end(), 0);
        A(thrust::raw_pointer_cast(&p[0]), thrust::raw_pointer_cast(&y[0]));  

        // alpha <- <r,r>/<Ap,p>
        ValueType alpha =  r2 / blas::dot(p.begin(), p.end(), y.begin());
        // x <- x + alpha * p
        blas::axpy(p.begin(), p.end(), x.begin(), alpha);
        // r <- r - alpha * Ap		
        blas::axpy(y.begin(), y.end(), r.begin(), -alpha);
		
        // beta <- <r_{i+1},r_{i+1}>/<r,r> 
        ValueType r2_old = r2;
        r2 = blas::nrm2(r.begin(), r.end()); r2 *= r2;
        ValueType beta = r2 / r2_old;                       
		
        // p <- r + beta*p
        blas::scal(p.begin(), p.end(), beta);
        blas::axpy(r.begin(), r.end(), p.begin(), static_cast<ValueType>(1.0));
    }
	

    //cudaThreadSynchronize();

    // MFLOPs excludes BLAS operations
    //double elapsed = ((double) (clock() - start)) / CLOCKS_PER_SEC;
    //double MFLOPs = 2* ((double) i * (double) A.num_entries)/ (1e6 * elapsed);
    //printf("-iteration completed in %lfms  ( > %6.2lf MFLOPs )\n",1000*elapsed, MFLOPs );

    if (verbose)
    {
        ValueType r_rel = sqrt(r2) / r_norm; // relative residual
        if(r2 <= stop_tol)
            std::cout << "[CG] converged to " << r_rel << " relative residual in " << i << " iterations" << std::endl;
        else
            std::cout << "[CG] failed to converge within " << i << " iterations (achieved " << r_rel << " relative residual)" << std::endl;;
    }

}

} // end namespace krylov
} // end namespace cusp

