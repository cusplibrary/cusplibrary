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

#include <cusp/detail/type_traits.h>
#include <cusp/system/detail/adl/transpose.h>
#include <cusp/system/detail/generic/transpose.h>

namespace cusp
{
namespace detail
{

template<typename MatrixType1, typename MatrixType2>
struct is_transposeable
{
  typedef typename MatrixType1::format format1;
  typedef typename MatrixType2::format format2;

  typedef typename thrust::detail::eval_if<
      thrust::detail::is_same<format1,format2>::value,
          thrust::detail::or_< is_coo<MatrixType1>, is_csr<MatrixType1>, is_array2d<MatrixType1> >,
          thrust::detail::identity_<thrust::detail::false_type>
      >::type type;
};

template <typename DerivedPolicy, typename MatrixType1, typename MatrixType2>
void transpose(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
               const MatrixType1& A, MatrixType2& At,
               thrust::detail::true_type)
{
    using cusp::system::detail::generic::transpose;

    typename MatrixType1::format format;

    transpose(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, At, format);
}

template <typename DerivedPolicy, typename MatrixType1, typename MatrixType2>
void transpose(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
               const MatrixType1& A, MatrixType2& At,
               thrust::detail::false_type)
{
    typename as_csr_type<MatrixType1>::type A_csr(A);
    typename as_csr_type<MatrixType2>::type At_csr;
    cusp::transpose(exec, A_csr, At_csr);

    cusp::convert(At_csr, At);
}

} // end namespace detail

template <typename DerivedPolicy, typename MatrixType1, typename MatrixType2>
void transpose(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
               const MatrixType1& A, MatrixType2& At)
{
    typename detail::is_transposeable<MatrixType1,MatrixType2>::type transposeable;
    cusp::detail::transpose(exec, A, At, transposeable);
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

