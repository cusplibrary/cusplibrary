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

template<typename MatrixType1, typename MatrixType2, typename MatrixType3>
struct is_elementwiseable
{
  typedef typename MatrixType1::format format1;
  typedef typename MatrixType2::format format2;
  typedef typename MatrixType3::format format3;

  typedef typename thrust::detail::eval_if<
      thrust::detail::and_<
          thrust::detail::is_same<format1,format2>,
          thrust::detail::is_same<format2,format3> >::value,
            // thrust::detail::or_< is_coo<MatrixType1>, is_csr<MatrixType1>, is_array2d<MatrixType1> >,
            thrust::detail::or_< is_coo<MatrixType1>, is_array2d<MatrixType1> >,
          thrust::detail::identity_<thrust::detail::false_type>
      >::type type;
};

template <typename DerivedPolicy,
          typename MatrixType1, typename MatrixType2, typename MatrixType3,
          typename BinaryFunction>
void elementwise(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                 const MatrixType1& A, const MatrixType2& B, MatrixType3& C,
                 BinaryFunction op,
                 thrust::detail::true_type)
{
    using cusp::system::detail::generic::elementwise;

    typename MatrixType1::format format;

    elementwise(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, B, C, op, format);
}

template <typename DerivedPolicy,
          typename MatrixType1, typename MatrixType2, typename MatrixType3,
          typename BinaryFunction>
void elementwise(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                 const MatrixType1& A, const MatrixType2& B, MatrixType3& C,
                 BinaryFunction op,
                 thrust::detail::false_type)
{
    typename as_coo_type<MatrixType1>::type A_coo(A);
    typename as_coo_type<MatrixType2>::type B_coo(B);
    typename as_coo_type<MatrixType3>::type C_coo;

    cusp::elementwise(exec, A_coo, B_coo, C_coo, op);

    cusp::convert(C_coo, C);
}

} // end namespace detail

template <typename DerivedPolicy,
          typename MatrixType1, typename MatrixType2, typename MatrixType3,
          typename BinaryFunction>
void elementwise(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                 const MatrixType1& A, const MatrixType2& B, MatrixType3& C,
                 BinaryFunction op)
{
    if(A.num_rows != B.num_rows || A.num_cols != B.num_cols)
        throw cusp::invalid_input_exception("matrix dimensions do not match");

    typename detail::is_elementwiseable<MatrixType1,MatrixType2,MatrixType3>::type is_elementwiseable;
    cusp::detail::elementwise(exec, A, B, C, op, is_elementwiseable);
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

