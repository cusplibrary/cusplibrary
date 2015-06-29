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
namespace cg_detail
{

template <typename DerivedPolicy,
          class LinearOperator,
          class Vector,
          class Monitor,
          class Preconditioner>
void cg(thrust::execution_policy<DerivedPolicy> &exec,
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

    // allocate workspace
    cusp::array1d<ValueType,MemorySpace> y(N);
    cusp::array1d<ValueType,MemorySpace> z(N);
    cusp::array1d<ValueType,MemorySpace> r(N);
    cusp::array1d<ValueType,MemorySpace> p(N);

    assert(A.num_rows == A.num_cols);        // sanity check

    // y <- Ax
    cusp::multiply(exec, A, x, y);

    // r <- b - A*x
    cusp::blas::axpby(exec, b, y, r, ValueType(1), ValueType(-1));

    // z <- M*r
    cusp::multiply(exec, M, r, z);

    // p <- z
    blas::copy(exec, z, p);

    // rz = <r^H, z>
    ValueType rz = blas::dotc(exec, r, z);

    while (!monitor.finished(r))
    {
        // y <- Ap
        cusp::multiply(exec, A, p, y);

        // alpha <- <r,z>/<y,p>
        ValueType alpha =  rz / blas::dotc(exec, y, p);

        // x <- x + alpha * p
        blas::axpy(exec, p, x, alpha);

        // r <- r - alpha * y
        blas::axpy(exec, y, r, -alpha);

        // z <- M*r
        cusp::multiply(exec, M, r, z);

        ValueType rz_old = rz;

        // rz = <r^H, z>
        rz = blas::dotc(exec, r, z);

        // beta <- <r_{i+1},r_{i+1}>/<r,r>
        ValueType beta = rz / rz_old;

        // p <- r + beta*p
        blas::axpby(exec, z, p, p, ValueType(1), beta);

        ++monitor;
    }
}

} // end cg_detail namespace

template <typename DerivedPolicy,
          class LinearOperator,
          class Vector>
void cg(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
        LinearOperator& A,
        Vector& x,
        Vector& b)
{
    typedef typename LinearOperator::value_type   ValueType;

    cusp::monitor<ValueType> monitor(b);

    cusp::krylov::cg(exec, A, x, b, monitor);
}

template <class LinearOperator,
          class Vector>
void cg(LinearOperator& A,
        Vector& x,
        Vector& b)
{
    using thrust::system::detail::generic::select_system;

    typedef typename LinearOperator::memory_space System1;
    typedef typename Vector::memory_space         System2;

    System1 system1;
    System2 system2;

    cusp::krylov::cg(select_system(system1,system2), A, x, b);
}

template <typename DerivedPolicy,
          class LinearOperator,
          class Vector,
          class Monitor>
void cg(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
        LinearOperator& A,
        Vector& x,
        Vector& b,
        Monitor& monitor)
{
    typedef typename LinearOperator::value_type   ValueType;
    typedef typename LinearOperator::memory_space MemorySpace;

    cusp::identity_operator<ValueType,MemorySpace> M(A.num_rows, A.num_cols);

    cusp::krylov::cg(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, x, b, monitor, M);
}

template <class LinearOperator,
          class Vector,
          class Monitor>
void cg(LinearOperator& A,
        Vector& x,
        Vector& b,
        Monitor& monitor)
{
    using thrust::system::detail::generic::select_system;

    typedef typename LinearOperator::memory_space System1;
    typedef typename Vector::memory_space         System2;

    System1 system1;
    System2 system2;

    cusp::krylov::cg(select_system(system1,system2), A, x, b, monitor);
}

template <typename DerivedPolicy,
          class LinearOperator,
          class Vector,
          class Monitor,
          class Preconditioner>
void cg(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
        LinearOperator& A,
        Vector& x,
        Vector& b,
        Monitor& monitor,
        Preconditioner& M)
{
    cusp::krylov::cg_detail::cg(thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
                                A, x, b, monitor, M);
}

template <class LinearOperator,
          class Vector,
          class Monitor,
          class Preconditioner>
void cg(LinearOperator& A,
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

    cusp::krylov::cg(select_system(system1,system2), A, x, b, monitor, M);
}

} // end namespace krylov
} // end namespace cusp

