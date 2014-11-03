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

/*! \file transpose.inl
 *  \brief Inline file for transpose.h.
 */

#include <thrust/detail/config.h>
#include <thrust/system/detail/generic/select_system.h>

#include <cusp/transpose.h>
#include <cusp/system/detail/adl/transpose.h>
#include <cusp/system/detail/generic/transpose.h>

namespace cusp
{
namespace detail
{

template <typename DerivedPolicy, typename MatrixType1, typename MatrixType2, typename MatrixFormat>
void transpose(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
               const MatrixType1& A, MatrixType2& At,
               MatrixFormat format1, MatrixFormat format2)
{
    using cusp::system::detail::generic::transpose;

    transpose(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, At, format1);
}

template <typename DerivedPolicy, typename MatrixType1, typename MatrixType2,
          typename MatrixFormat1, typename MatrixFormat2>
void transpose(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
               const MatrixType1& A, MatrixType2& At,
               MatrixFormat1 format1, MatrixFormat2 format2)
{
    typedef typename MatrixType1::index_type   IndexType1;
    typedef typename MatrixType1::value_type   ValueType1;
    typedef typename MatrixType1::memory_space MemorySpace1;

    typedef typename MatrixType2::index_type   IndexType2;
    typedef typename MatrixType2::value_type   ValueType2;
    typedef typename MatrixType2::memory_space MemorySpace2;

    cusp::csr_matrix<IndexType1, ValueType1, MemorySpace1> A_csr(A);
    cusp::csr_matrix<IndexType2, ValueType2, MemorySpace2> At_csr;
    cusp::transpose(exec, A_csr, At_csr);

    cusp::convert(At_csr, At);
}

} // end namespace detail

template <typename DerivedPolicy, typename MatrixType1, typename MatrixType2>
void transpose(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
               const MatrixType1& A, MatrixType2& At)
{
    typedef typename MatrixType1::format Format1;
    typedef typename MatrixType2::format Format2;

    Format1 format1;
    Format2 format2;

    cusp::detail::transpose(exec, A, At, format1, format2);
}

template <typename MatrixType1, typename MatrixType2>
void transpose(const MatrixType1& A, MatrixType2& At)
{
    using thrust::system::detail::generic::select_system;

    typedef typename MatrixType1::memory_space System1;
    typedef typename MatrixType2::memory_space System2;

    System1 system1;
    System2 system2;

    cusp::transpose(select_system(system1,system2), A, At);
}

} // end namespace cusp

