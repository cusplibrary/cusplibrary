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

/*! \file elementwise.inl
 *  \brief Inline file for elementwise.h.
 */

#include <thrust/detail/config.h>
#include <thrust/system/detail/generic/select_system.h>

#include <cusp/elementwise.h>
#include <cusp/system/detail/adl/elementwise.h>
#include <cusp/system/detail/generic/elementwise.h>

namespace cusp
{
namespace detail
{

template <typename DerivedPolicy,
          typename MatrixType1, typename MatrixType2, typename MatrixType3,
          typename BinaryFunction, typename MatrixFormat>
void elementwise(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                 const MatrixType1& A, const MatrixType2& B, MatrixType3& C,
                 BinaryFunction func,
                 MatrixFormat format1, MatrixFormat format2, MatrixFormat format3)
{
    using cusp::system::detail::generic::elementwise;

    elementwise(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, B, C, func, format1);
}

template <typename DerivedPolicy,
          typename MatrixType1, typename MatrixType2, typename MatrixType3,
          typename BinaryFunction,
          typename MatrixFormat1, typename MatrixFormat2, typename MatrixFormat3>
void elementwise(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                 const MatrixType1& A, const MatrixType2& B, MatrixType3& C,
                 BinaryFunction func,
                 MatrixFormat1 format1, MatrixFormat2 format2, MatrixFormat3 format3)
{
    typedef typename MatrixType1::index_type   IndexType1;
    typedef typename MatrixType1::value_type   ValueType1;
    typedef typename MatrixType1::memory_space MemorySpace1;

    typedef typename MatrixType2::index_type   IndexType2;
    typedef typename MatrixType2::value_type   ValueType2;
    typedef typename MatrixType2::memory_space MemorySpace2;

    cusp::csr_matrix<IndexType1, ValueType1, MemorySpace1> A_csr(A);
    cusp::csr_matrix<IndexType2, ValueType2, MemorySpace2> B_csr(B);
    cusp::csr_matrix<IndexType2, ValueType2, MemorySpace2> C_csr;

    cusp::elementwise(exec, A_csr, B_csr, C_csr, func);

    cusp::convert(C_csr, C);
}

} // end namespace detail

template <typename DerivedPolicy,
          typename MatrixType1, typename MatrixType2, typename MatrixType3,
          typename BinaryFunction>
void elementwise(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                 const MatrixType1& A, const MatrixType2& B, MatrixType3& C,
                 BinaryFunction func)
{
    typedef typename MatrixType1::format Format1;
    typedef typename MatrixType2::format Format2;

    cusp::assert_same_dimensions(A,B);

    Format1 format1;
    Format2 format2;

    cusp::detail::elementwise(exec, A, B, C, func, format1, format2);
}

template <typename MatrixType1, typename MatrixType2, typename MatrixType3>
void add(const MatrixType1& A, const MatrixType2& B, MatrixType3& C)
{
    using thrust::system::detail::generic::select_system;

    typedef typename MatrixType1::value_type   ValueType;
    typedef thrust::plus<ValueType>            Op;

    typedef typename MatrixType1::memory_space System1;
    typedef typename MatrixType2::memory_space System2;

    Op op;
    System1 system1;
    System2 system2;

    cusp::elementwise(select_system(system1,system2), A, B, C, op);
}

template <typename MatrixType1, typename MatrixType2, typename MatrixType3>
void subtract(const MatrixType1& A, const MatrixType2& B, MatrixType3& C)
{
    using thrust::system::detail::generic::select_system;

    typedef typename MatrixType1::value_type   ValueType;
    typedef thrust::minus<ValueType>           Op;

    typedef typename MatrixType1::memory_space System1;
    typedef typename MatrixType2::memory_space System2;

    Op op;
    System1 system1;
    System2 system2;

    cusp::elementwise(select_system(system1,system2), A, B, C, op);
}

} // end namespace cusp

