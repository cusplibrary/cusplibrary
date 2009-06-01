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

#include <cusp/support/allocate.h>
#include <cusp/support/copy.h>

namespace cusp
{
    template <typename T>
    T * duplicate_array_to_host(const T * d_ptr, const size_t N)
    {
        T * h_ptr = new_host_array<T>(N);
        memcpy_to_host(h_ptr, d_ptr, N);
        return h_ptr;
    }
    
    template <typename T>
    T * duplicate_array_to_device(const T * h_ptr, const size_t N)
    {
        T * d_ptr = new_device_array<T>(N);
        memcpy_to_device(d_ptr, h_ptr, N);
        return d_ptr;
    }
    
    template <typename T>
    T * duplicate_array_on_host(const T * h_src, const size_t N)
    {
        T * h_dst = new_host_array<T>(N);
        memcpy_on_host(h_dst, h_src, N);
        return h_dst;
    }
    
    template <typename T>
    T *  duplicate_array_on_device(const T * d_src, const size_t N)
    {
        T * d_dst = new_device_array<T>(N);
        memcpy_on_device(d_dst, d_src, N);
        return d_dst;
    }
    
} // end namespace cusp
