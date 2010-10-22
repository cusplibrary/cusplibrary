/*
 *  Copyright 2008-2009 NVIDIA Corporation
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

#include <memory>

#include <thrust/device_allocator.h>
#include <thrust/iterator/iterator_traits.h>

namespace cusp
{
  typedef thrust::host_space_tag                host_memory;
  typedef thrust::detail::cuda_device_space_tag device_memory;
  
   template<typename T, typename MemorySpace>
   struct default_memory_allocator
      : thrust::detail::eval_if<
          thrust::detail::is_convertible<MemorySpace, host_memory>::value,
  
          thrust::detail::identity_< std::allocator<T> >,
  
          // XXX add backend-specific allocators here?
  
          thrust::detail::eval_if<
            thrust::detail::is_convertible<MemorySpace, device_memory>::value,
  
            thrust::detail::identity_< thrust::device_malloc_allocator<T> >,
  
            thrust::detail::identity_< MemorySpace >
          >
        >
  {};
  
} // end namespace cusp

