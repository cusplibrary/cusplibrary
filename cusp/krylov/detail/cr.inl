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


#include <cusp/array1d.h>
#include <cusp/blas/blas.h>
#include <cusp/multiply.h>
#include <cusp/monitor.h>
#include <cusp/linear_operator.h>

namespace blas = cusp::blas;

namespace cusp
{
namespace krylov
{

template <class LinearOperator,
         class Vector>
void cr(LinearOperator& A,
        Vector& x,
        Vector& b)
{
    typedef typename LinearOperator::value_type   ValueType;

    cusp::monitor<ValueType> monitor(b);

    cusp::krylov::cr(A, x, b, monitor);
}

template <class LinearOperator,
         class Vector,
         class Monitor>
void cr(LinearOperator& A,
        Vector& x,
        Vector& b,
        Monitor& monitor)
{
    typedef typename LinearOperator::value_type   ValueType;
    typedef typename LinearOperator::memory_space MemorySpace;

    cusp::identity_operator<ValueType,MemorySpace> M(A.num_rows, A.num_cols);

    cusp::krylov::cr(A, x, b, monitor, M);
}

template <class LinearOperator,
         class Vector,
         class Monitor,
         class Preconditioner>
void cr(LinearOperator& A,
        Vector& x,
        Vector& b,
        Monitor& monitor,
        Preconditioner& M)
{
    typedef typename LinearOperator::value_type   ValueType;
    typedef typename LinearOperator::memory_space MemorySpace;

    assert(A.num_rows == A.num_cols);        // sanity check

    const size_t N = A.num_rows;
    const size_t recompute_r = 8;	     // interval to update r

    // allocate workspace
    cusp::array1d<ValueType,MemorySpace> y(N);
    cusp::array1d<ValueType,MemorySpace> z(N);
    cusp::array1d<ValueType,MemorySpace> r(N);
    cusp::array1d<ValueType,MemorySpace> p(N);
    cusp::array1d<ValueType,MemorySpace> Az(N);
    cusp::array1d<ValueType,MemorySpace> Ax(N);

    // y <- A*x
    cusp::multiply(A, x, Ax);

    // r <- b - A*x
    blas::axpby(b, Ax, r, ValueType(1), ValueType(-1));

    // z <- M*r
    cusp::multiply(M, r, z);

    // p <- z
    blas::copy(z, p);

    // y <- A*p
    cusp::multiply(A, p, y);

    // Az <- A*z
    cusp::multiply(A, z, Az);

    // rz = <r^H, z>
    ValueType rz = blas::dotc(r, Az);

    while (!monitor.finished(r))
    {
        // alpha <- <r,z>/<y,p>
        ValueType alpha =  rz / blas::dotc(y, y);

        // x <- x + alpha * p
        blas::axpy(p, x, alpha);

        size_t iter = monitor.iteration_count();
        if( (iter % recompute_r) && (iter > 0) )
        {
            // r <- r - alpha * y
            blas::axpy(y, r, -alpha);
        }
        else
        {
            // y <- A*x
            cusp::multiply(A, x, Ax);

            // r <- b - A*x
            blas::axpby(b, Ax, r, ValueType(1), ValueType(-1));
        }

        // z <- M*r
        cusp::multiply(M, r, z);

    	// Az <- A*z
    	cusp::multiply(A, z, Az);

        ValueType rz_old = rz;

        // rz = <r^H, z>
        rz = blas::dotc(r, Az);

        // beta <- <r_{i+1},r_{i+1}>/<r,r>
        ValueType beta = rz / rz_old;

        // p <- r + beta*p
        blas::axpby(z, p, p, ValueType(1), beta);

        // y <- z + beta*p
        blas::axpby(Az, y, y, ValueType(1), beta);

        ++monitor;
    }
}

} // end namespace krylov
} // end namespace cusp

