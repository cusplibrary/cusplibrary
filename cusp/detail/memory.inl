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

#include <cusp/iterator/iterator_traits.h>

#include <thrust/device_allocator.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/iterator/detail/minimum_system.h>
#include <thrust/iterator/iterator_traits.h>

#include <cuda/std/type_traits>
#include <memory>

namespace cusp
{

template<typename T, typename MemorySpace>
struct default_memory_allocator
        : ::cuda::std::conditional<
        ::cuda::std::is_same<MemorySpace, host_memory>::value,
        std::allocator<T>,
        thrust::device_malloc_allocator<T>
        >
{};

template <typename MemorySpace1, typename MemorySpace2, typename MemorySpace3, typename MemorySpace4>
struct minimum_space
{
    typedef thrust::detail::minimum_system_t<MemorySpace1, MemorySpace2, MemorySpace3, MemorySpace4> type;
};

} // end namespace cusp

