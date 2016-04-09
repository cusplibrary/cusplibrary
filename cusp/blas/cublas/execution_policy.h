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

#include <cublas_v2.h>

#include <cusp/detail/config.h>
#include <cusp/blas/blas_policy.h>

#include <thrust/system/cuda/detail/execution_policy.h>

namespace cusp
{
namespace blas
{
namespace cublas
{

class execution_policy
  : public thrust::system::cuda::detail::execution_policy<execution_policy>
{
  public:

    execution_policy(void) {}

    execution_policy(const cublasHandle_t& handle)
      : handle(handle)
    {}

    thrust::system::cuda::detail::execution_policy<execution_policy>
    with_cublas(const cublasHandle_t &h) const
    {
      // create a copy of *this to return
      // make sure it is the derived type
      execution_policy result = *this;

      // change the result's cublas handle to h
      result.set_handle(h);

      return result;
    }

    inline cublasHandle_t& get_handle(void)
    {
      return handle;
    }

  private:

    inline void set_handle(const cublasHandle_t &h)
    {
      handle = h;
    }

    cublasHandle_t handle;
};

} // end cublas
} // end blas

// alias items at top-level
namespace cublas
{

using cusp::blas::cublas::execution_policy;

} // end cublas
} // end cusp

