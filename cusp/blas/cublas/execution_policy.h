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
#include <cusp/blas/blas_policy.h>

#include <cublas_v2.h>
#include <thrust/system/cuda/detail/execution_policy.h>

namespace cusp
{
namespace blas
{
namespace cublas
{
namespace detail
{

template<typename DerivedPolicy>
class execution_policy
  : public thrust::system::cuda::detail::execution_policy<DerivedPolicy>
{
  public:

    execution_policy<DerivedPolicy> with_cublas(const cublasHandle_t &h) const
    {
      // create a copy of *this to return
      // make sure it is the derived type
      execution_policy<DerivedPolicy> result = *this;

      // change the result's cublas handle to h
      result.set_cublas_handle(h);

      return result;
    }

    inline cublasHandle_t handle(void)
    {
      return cublasHandle;
    }

  private:

    __host__ __device__
    inline void set_cublas_handle(const cublasHandle_t &h)
    {
      cublasHandle = h;
    }

    cublasHandle_t cublasHandle;
};

} // end detail

// alias execution_policy and tag here
using cusp::blas::cublas::detail::execution_policy;

} // end cublas
} // end blas

// alias items at top-level
namespace cublas
{

using cusp::blas::cublas::execution_policy;

} // end cublas
} // end cusp

