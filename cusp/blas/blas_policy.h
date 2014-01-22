/*
 *  Copyright 2008-2012 NVIDIA Corporation
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

#define __CUSP_HOST_BLAS_POLICY_HEADER <__CUSP_HOST_BLAS_ROOT/blas_policy.h>
#include __CUSP_HOST_BLAS_POLICY_HEADER
#undef __CUSP_HOST_BLAS_POLICY_HEADER

#define __CUSP_DEVICE_BLAS_POLICY_HEADER <__CUSP_DEVICE_BLAS_ROOT/blas_policy.h>
#include __CUSP_DEVICE_BLAS_POLICY_HEADER
#undef __CUSP_DEVICE_BLAS_POLICY_HEADER

#include <cusp/blas/thrustblas/blas_policy.h>

namespace cusp
{
namespace blas
{

/// helper classes [4.3].
template<typename _Tp, _Tp __v>
  struct integral_constant
  {
    static const _Tp                      value = __v;
    typedef _Tp                           value_type;
    typedef integral_constant<_Tp, __v>   type;
  };

/// typedef for true_type
typedef integral_constant<bool, true>     true_type;

/// typedef for true_type
typedef integral_constant<bool, false>    false_type;

template<typename T> struct is_floating_point                                : public false_type {};
template<>           struct is_floating_point<float>                         : public true_type {};
template<>           struct is_floating_point<double>                        : public true_type {};
template<>           struct is_floating_point< cusp::complex<float>  >       : public true_type {};
template<>           struct is_floating_point< cusp::complex<double> >       : public true_type {};

template<typename T>
struct type_wrapper
{
  typedef T type;
};

template<typename ValueType, typename MemorySpace>
struct blas_policy
{
    typedef cusp::blas::thrustblas::blas_policy<MemorySpace> type;
};

template<typename ValueType>
struct blas_policy<ValueType,cusp::host_memory>
{
    typedef typename thrust::detail::eval_if<is_floating_point<ValueType>::value,
        type_wrapper< cusp::blas::__CUSP_HOST_BLAS_NAMESPACE::blas_policy<cusp::host_memory> >,
        type_wrapper< cusp::blas::thrustblas::blas_policy<cusp::host_memory> >
        >::type type;
};

template<typename ValueType>
struct blas_policy<ValueType,cusp::device_memory>
{
    typedef typename thrust::detail::eval_if<is_floating_point<ValueType>::value,
        type_wrapper< cusp::blas::__CUSP_DEVICE_BLAS_NAMESPACE::blas_policy<cusp::device_memory> >,
        type_wrapper< cusp::blas::thrustblas::blas_policy<cusp::device_memory> >
        >::type type;
};

} // end blas
} // end cusp

#define __CUSP_HOST_BLAS_SYSTEM <__CUSP_HOST_BLAS_ROOT/blas.h>
#include __CUSP_HOST_BLAS_SYSTEM
#undef __CUSP_HOST_BLAS_SYSTEM

#define __CUSP_DEVICE_BLAS_SYSTEM <__CUSP_DEVICE_BLAS_ROOT/blas.h>
#include __CUSP_DEVICE_BLAS_SYSTEM
#undef __CUSP_DEVICE_BLAS_SYSTEM

#include <cusp/blas/thrustblas/blas.h>

