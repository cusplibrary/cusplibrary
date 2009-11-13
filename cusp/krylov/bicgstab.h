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

#include <cusp/array1d.h>
#include <cusp/blas.h>
#include <cusp/spblas.h>

#include <iostream>

namespace blas = cusp::blas;

namespace cusp
{
namespace krylov
{

template <class LinearOperator,
          class Vector>
void bicgstab(LinearOperator A,
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

    cusp::array1d<ValueType,MemorySpace>   p(N);
    cusp::array1d<ValueType,MemorySpace>   r(N);
    cusp::array1d<ValueType,MemorySpace>   r_star(N);
    cusp::array1d<ValueType,MemorySpace>   s(N);
    cusp::array1d<ValueType,MemorySpace>  Mp(N);
    cusp::array1d<ValueType,MemorySpace> AMp(N);
    cusp::array1d<ValueType,MemorySpace>  Ms(N);
    cusp::array1d<ValueType,MemorySpace> AMs(N);

    // TODO fuse r = b - A*x into a single SpMV

    // y <- Ax
    blas::fill(y, 0);
    cusp::spblas::spmv(A, x, y);

    // r <- b - A*x
    blas::axpby(b, y, r, static_cast<ValueType>(1.0), static_cast<ValueType>(-1.0));

    // p <- r
    blas::copy(r, p);

    // r_star <- r
    blas::copy(r, r_star);

    // r_norm < || r ||
    ValueType r_norm_initial = blas::nrm2(r);
    ValueType r_norm         = r_norm_initial;

    ValueType r_r_star_old = blas::dotc(r_star, r);

    if (verbose)
        std::cout << "[BiCGstab] initial residual norm " << r_norm << std::endl;

    ValueType stop_tol = r_norm_initial * tol;

    size_t i = 0;

    while( i++ < max_iter && r_norm > stop_tol )
    {
        // Mp = M*p
        blas::copy(p, Mp); // TODO apply preconditioner, identity for now

        // AMp = A*Mp
        blas::fill(AMp, 0);
        cusp::spblas::spmv(A, Mp, AMp);

        // alpha = (r_j, r_star) / (A*M*p, r_star)
        ValueType alpha = r_r_star_old / blas::dotc(r_star, AMp);
        
        // s_j = r_j - alpha * AMp
        blas::axpby(r, AMp, s, static_cast<ValueType>(1.0), static_cast<ValueType>(-alpha));

        // Ms = M*s_j
        blas::copy(s, Ms); // TODO apply preconditioner, identity for now
        
        // AMs = A*Ms
        blas::fill(AMs, 0);
        cusp::spblas::spmv(A, Ms, AMs);

        // omega = (AMs, s) / (AMs, AMs)
        ValueType omega = blas::dotc(AMs, s) / blas::dotc(AMs, AMs); // TODO optimize denominator
        
        // x_{j+1} = x_j + alpha*M*p_j + omega*M*s_j
        blas::axpbypcz(x, Mp, Ms, x, static_cast<ValueType>(1.0), alpha, omega);

        // r_{j+1} = s_j - omega*A*M*s
        blas::axpby(s, AMs, r, static_cast<ValueType>(1.0), -omega);

        // beta_j = (r_{j+1}, r_star) / (r_j, r_star) * (alpha/omega)
        ValueType r_r_star_new = blas::dotc(r_star, r);
        ValueType beta = (r_r_star_new / r_r_star_old) * (alpha / omega);
        r_r_star_old = r_r_star_new;

        // p_{j+1} = r_{j+1} + beta*(p_j - omega*A*M*p)
        blas::axpbypcz(r, p, AMp, p, static_cast<ValueType>(1.0), beta, -beta*omega);

        i++;

        r_norm = blas::nrm2(r);
    }


    //cudaThreadSynchronize();

    // MFLOPs excludes BLAS operations
    //double elapsed = ((double) (clock() - start)) / CLOCKS_PER_SEC;
    //double MFLOPs = 2* ((double) i * (double) A.num_entries)/ (1e6 * elapsed);
    //printf("-iteration completed in %lfms  ( > %6.2lf MFLOPs )\n",1000*elapsed, MFLOPs );

    if (verbose)
    {
        ValueType r_rel = r_norm / r_norm_initial; // relative residual
        if(r_norm <= stop_tol)
            std::cout << "[BiCGstab] converged to " << r_rel << " relative residual in " << i << " iterations" << std::endl;
        else
            std::cout << "[BiCGstab] failed to converge within " << i << " iterations (achieved " << r_rel << " relative residual)" << std::endl;;
    }
}

} // end namespace krylov
} // end namespace cusp

