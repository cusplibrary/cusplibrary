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

#include <cusp/detail/type_traits.h>
#include <cusp/system/detail/adl/elementwise.h>
#include <cusp/system/detail/generic/elementwise.h>

namespace cusp
{
namespace detail
{

template <typename DerivedPolicy,
          typename MatrixType1, typename MatrixType2, typename MatrixType3,
          typename BinaryFunction, typename Format>
void elementwise(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                 const MatrixType1& A, const MatrixType2& B, MatrixType3& C,
                 BinaryFunction op,
                 Format, Format, Format)
{
    using cusp::system::detail::generic::elementwise;

    elementwise(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, B, C, op, Format());
}

template <typename DerivedPolicy,
          typename MatrixType1, typename MatrixType2, typename MatrixType3,
          typename BinaryFunction,
          typename Format1, typename Format2, typename Format3>
void elementwise(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                 const MatrixType1& A, const MatrixType2& B, MatrixType3& C,
                 BinaryFunction op,
                 Format1, Format2, Format3)
{
    typedef typename cusp::detail::coo_view_type<MatrixType1>::const_view View1;
    typedef typename cusp::detail::coo_view_type<MatrixType2>::const_view View2;
    typedef typename cusp::detail::as_coo_type<MatrixType3>::type CooMatrixType;

    View1 A_coo(A);
    View2 B_coo(B);
    CooMatrixType C_coo;

    cusp::elementwise(exec, A_coo, B_coo, C_coo, op);

    cusp::convert(exec, C_coo, C);
}

} // end namespace detail

template <typename DerivedPolicy,
          typename MatrixType1, typename MatrixType2, typename MatrixType3,
          typename BinaryFunction>
void elementwise(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                 const MatrixType1& A, const MatrixType2& B, MatrixType3& C,
                 BinaryFunction op)
{
    typename MatrixType1::format format1;
    typename MatrixType2::format format2;
    typename MatrixType3::format format3;

    if(A.num_rows != B.num_rows || A.num_cols != B.num_cols)
        throw cusp::invalid_input_exception("matrix dimensions do not match");

    cusp::detail::elementwise(exec, A, B, C, op, format1, format2, format3);
}

template <typename MatrixType1, typename MatrixType2, typename MatrixType3,
          typename BinaryFunction>
void elementwise(const MatrixType1& A, const MatrixType2& B, MatrixType3& C,
                 BinaryFunction op)
{
    using thrust::system::detail::generic::select_system;

    typedef typename MatrixType1::memory_space System1;
    typedef typename MatrixType2::memory_space System2;
    typedef typename MatrixType3::memory_space System3;

    System1 system1;
    System2 system2;
    System3 system3;

    cusp::elementwise(select_system(system1,system2,system3), A, B, C, op);
}

template <typename MatrixType1, typename MatrixType2, typename MatrixType3>
void add(const MatrixType1& A, const MatrixType2& B, MatrixType3& C)
{
    typedef typename MatrixType1::value_type   ValueType;
    typedef thrust::plus<ValueType>            Op;

    Op op;

    cusp::elementwise(A, B, C, op);
}

template <typename MatrixType1, typename MatrixType2, typename MatrixType3>
void subtract(const MatrixType1& A, const MatrixType2& B, MatrixType3& C)
{
    typedef typename MatrixType1::value_type   ValueType;
    typedef thrust::minus<ValueType>           Op;

    Op op;

    cusp::elementwise(A, B, C, op);
}

} // end namespace cusp

