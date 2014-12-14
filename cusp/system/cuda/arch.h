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

#pragma once

#include <cusp/detail/config.h>

#include <thrust/extrema.h>

#if THRUST_VERSION >= 100700
#include <thrust/system/cuda/detail/detail/launch_calculator.h>
#elif THRUST_VERSION >= 100600
#include <thrust/system/cuda/detail/arch.h>
#else
#include <thrust/detail/backend/cuda/arch.h>
#endif

namespace cusp
{
namespace system
{
namespace cuda
{

//maximum number of co-resident threads
const int MAX_THREADS = (30 * 1024);
const int WARP_SIZE = 32; 

namespace detail
{

template <typename KernelFunction>
size_t max_active_blocks(KernelFunction kernel, const size_t CTA_SIZE, const size_t dynamic_smem_bytes)
{
#if THRUST_VERSION >= 100700
  using namespace thrust::system::cuda::detail;
  function_attributes_t attributes = function_attributes(kernel);
  device_properties_t properties = device_properties();
  return properties.multiProcessorCount * cuda_launch_config_detail::max_active_blocks_per_multiprocessor(properties, attributes, CTA_SIZE, dynamic_smem_bytes);
#elif THRUST_VERSION >= 100600
  return thrust::system::cuda::detail::arch::max_active_blocks(kernel, CTA_SIZE, dynamic_smem_bytes);
#else
  return thrust::detail::backend::cuda::arch::max_active_blocks(kernel, CTA_SIZE, dynamic_smem_bytes);
#endif
}

} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace cusp

