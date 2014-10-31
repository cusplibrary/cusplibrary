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

template <typename DerivedPolicy, typename MatrixType1, typename MatrixType2, typename Format1, typename Format2>
void transpose(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
               const MatrixType1& A, MatrixType2& At, Format1& fmt1, Format2& fmt2)
{
    using cusp::system::detail::generic::transpose;
    transpose(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, At, fmt1, fmt2);
}

template <typename DerivedPolicy, typename MatrixType1, typename MatrixType2>
void transpose(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
               const MatrixType1& A, MatrixType2& At)
{
    using cusp::system::detail::generic::transpose;
    transpose(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), A, At,
              typename MatrixType1::format(), typename MatrixType2::format());
}

template <typename MatrixType1, typename MatrixType2>
void transpose(const MatrixType1& A, MatrixType2& At)
{
    using thrust::system::detail::generic::select_system;

    typedef typename MatrixType1::memory_space System;

    System system;

    std::cout << "Select system : " << typeid(System).name() << std::endl;

    cusp::transpose(select_system(system), A, At, typename MatrixType1::format(), typename MatrixType2::format());
}

} // end namespace cusp

