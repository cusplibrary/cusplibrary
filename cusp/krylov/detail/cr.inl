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
#include <cusp/multiply.h>
#include <cusp/monitor.h>
#include <cusp/linear_operator.h>

#include <cusp/blas/blas.h>

namespace blas = cusp::blas;

namespace cusp
{
namespace krylov
{
namespace cr_detail
{

template <typename DerivedPolicy,
          class LinearOperator,
          class Vector,
          class Monitor,
          class Preconditioner>
void cr(thrust::execution_policy<DerivedPolicy> &exec,
        LinearOperator& A,
        Vector& x,
        Vector& b,
        Monitor& monitor,
        Preconditioner& M)
{
    typedef typename LinearOperator::value_type           ValueType;
    typedef typename cusp::minimum_space<
    typename LinearOperator::memory_space,
             typename Vector::memory_space,
             typename Preconditioner::memory_space>::type  MemorySpace;

    assert(A.num_rows == A.num_cols);        // sanity check

    const size_t N = A.num_rows;
    const size_t recompute_r = 8;	     // interval to update r

    // allocate workspace
    cusp::detail::temporary_array<ValueType, DerivedPolicy> y(exec, N);
    cusp::detail::temporary_array<ValueType, DerivedPolicy> z(exec, N);
    cusp::detail::temporary_array<ValueType, DerivedPolicy> r(exec, N);
    cusp::detail::temporary_array<ValueType, DerivedPolicy> p(exec, N);
    cusp::detail::temporary_array<ValueType, DerivedPolicy> Az(exec, N);
    cusp::detail::temporary_array<ValueType, DerivedPolicy> Ax(exec, N);

    // y <- A*x
    cusp::multiply(exec, A, x, Ax);

    // r <- b - A*x
    blas::axpby(exec, b, Ax, r, ValueType(1), ValueType(-1));

    // z <- M*r
    cusp::multiply(exec, M, r, z);

    // p <- z
    blas::copy(exec, z, p);

    // y <- A*p
    cusp::multiply(exec, A, p, y);

    // Az <- A*z
    cusp::multiply(exec, A, z, Az);

    // rz = <r^H, z>
    ValueType rz = blas::dotc(exec, r, Az);

    while (!monitor.finished(r))
    {
        // alpha <- <r,z>/<y,p>
        ValueType alpha =  rz / blas::dotc(exec, y, y);

        // x <- x + alpha * p
        blas::axpy(exec, p, x, alpha);

        size_t iter = monitor.iteration_count();
        if( (iter % recompute_r) && (iter > 0) )
        {
            // r <- r - alpha * y
            blas::axpy(y, r, -alpha);
        }
        else
        {
            // y <- A*x
            cusp::multiply(exec, A, x, Ax);

            // r <- b - A*x
            blas::axpby(exec, b, Ax, r, ValueType(1), ValueType(-1));
        }

        // z <- M*r
        cusp::multiply(exec, M, r, z);

        // Az <- A*z
        cusp::multiply(exec, A, z, Az);

        ValueType rz_old = rz;

        // rz = <r^H, z>
        rz = blas::dotc(r, Az);

        // beta <- <r_{i+1},r_{i+1}>/<r,r>
        ValueType beta = rz / rz_old;

        // p <- r + beta*p
        blas::axpby(exec, z, p, p, ValueType(1), beta);

        // y <- z + beta*p
        blas::axpby(exec, Az, y, y, ValueType(1), beta);

        ++monitor;
    }
}

template <typename DerivedPolicy,
          class LinearOperator,
          class Vector,
          class Monitor>
void cr(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
        LinearOperator& A,
        Vector& x,
        Vector& b,
        Monitor& monitor)
{
    typedef typename LinearOperator::value_type   ValueType;
    typedef typename LinearOperator::memory_space MemorySpace;

    cusp::identity_operator<ValueType,MemorySpace> M(A.num_rows, A.num_cols);

    cusp::krylov::cr_detail::cr(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, x, b, monitor, M);
}

template <typename DerivedPolicy,
          class LinearOperator,
          class Vector>
void cr(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
        LinearOperator& A,
        Vector& x,
        Vector& b)
{
    typedef typename LinearOperator::value_type   ValueType;

    cusp::monitor<ValueType> monitor(b);

    cusp::krylov::cr_detail::cr(exec, A, x, b, monitor);
}

} // end cr_detail namespace

template <typename DerivedPolicy,
          class LinearOperator,
          class Vector>
void cr(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
        LinearOperator& A,
        Vector& x,
        Vector& b)
{
    using cusp::krylov::cr_detail::cr;

    cr(thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
       A, x, b);
}

template <class LinearOperator,
          class Vector>
void cr(LinearOperator& A,
        Vector& x,
        Vector& b)
{
    using thrust::system::detail::generic::select_system;

    typedef typename LinearOperator::memory_space System1;
    typedef typename Vector::memory_space         System2;

    System1 system1;
    System2 system2;

    cusp::krylov::cr(select_system(system1,system2), A, x, b);
}

template <typename DerivedPolicy,
          class LinearOperator,
          class Vector,
          class Monitor>
void cr(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
        LinearOperator& A,
        Vector& x,
        Vector& b,
        Monitor& monitor)
{
    using cusp::krylov::cr_detail::cr;

    cr(thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
       A, x, b, monitor);
}

template <class LinearOperator,
          class Vector,
          class Monitor>
void cr(LinearOperator& A,
        Vector& x,
        Vector& b,
        Monitor& monitor)
{
    using thrust::system::detail::generic::select_system;

    typedef typename LinearOperator::memory_space System1;
    typedef typename Vector::memory_space         System2;

    System1 system1;
    System2 system2;

    cusp::krylov::cr(select_system(system1,system2), A, x, b, monitor);
}

template <typename DerivedPolicy,
          class LinearOperator,
          class Vector,
          class Monitor,
          class Preconditioner>
void cr(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
        LinearOperator& A,
        Vector& x,
        Vector& b,
        Monitor& monitor,
        Preconditioner& M)
{
    using cusp::krylov::cr_detail::cr;

    cr(thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
       A, x, b, monitor, M);
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
    using thrust::system::detail::generic::select_system;

    typedef typename LinearOperator::memory_space System1;
    typedef typename Vector::memory_space         System2;

    System1 system1;
    System2 system2;

    cusp::krylov::cr(select_system(system1,system2), A, x, b, monitor, M);
}

} // end namespace krylov
} // end namespace cusp

