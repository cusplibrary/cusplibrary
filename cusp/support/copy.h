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

#include <cuda_runtime_api.h>

namespace cusp
{
    namespace detail
    {
        template<cudaMemcpyKind Kind>
        struct cuda_copier
        {
            template<typename T>
            void copy(T *dst, const T *src, size_t count) const
            {
                cudaMemcpy(dst, src, sizeof(T)*count, Kind);
            }
        };
    } // end namespace detail

    template<class DstMemorySpace, class SrcMemorySpace>  
    struct memory_transfer {};
    template<>
    struct memory_transfer<host_memory,host_memory>     : detail::cuda_copier<cudaMemcpyHostToHost>     {};
    template<>
    struct memory_transfer<device_memory,host_memory>   : detail::cuda_copier<cudaMemcpyHostToDevice>   {};
    template<>
    struct memory_transfer<host_memory,device_memory>   : detail::cuda_copier<cudaMemcpyDeviceToHost>   {};
    template<>
    struct memory_transfer<device_memory,device_memory> : detail::cuda_copier<cudaMemcpyDeviceToDevice> {};

    struct host_to_host     : public memory_transfer<host_memory, host_memory> {};
    struct host_to_device   : public memory_transfer<device_memory, host_memory> {};
    struct device_to_host   : public memory_transfer<host_memory, device_memory> {};
    struct device_to_device : public memory_transfer<device_memory, device_memory> {};

   
    template<typename T, class Copy>
    void memcpy_array(T *dst, const T *src, size_t count, Copy copier = Copy())
    {
        copier.copy(dst, src, count);
    }
    
    template<typename T, class MemorySpace1, class MemorySpace2>
    void memcpy_array(T *dst, const T *src, size_t count)
    {
        memory_transfer<MemorySpace1,MemorySpace2>().copy(dst, src, count);
    }

    ///////////////////////
    // Predefined Copies //
    ///////////////////////

    template<class T>
    void memcpy_on_host(T *h_dst, const T *h_src, size_t N=1)
    {
        memcpy_array(h_dst, h_src, N, host_to_host());
    }

    template<class T>
    void memcpy_to_device(T *d_dst, const T *h_src, size_t N=1)
    {
        memcpy_array(d_dst, h_src, N, host_to_device());
    }

    template<class T>
    void memcpy_to_host(T *h_dst, const T *d_src, size_t N=1)
    {
        memcpy_array(h_dst, d_src, N, device_to_host());
    }

    template<class T>
    void memcpy_on_device(T *d_dst, const T *d_src, size_t N=1)
    {
        memcpy_array(d_dst, d_src, N, device_to_device());
    }

    template<class T, class S>
    void memcpy_to_symbol(const S& s, const T *hp, size_t N=1)
    {
        cudaMemcpyToSymbol(s, hp, sizeof(T)*N);
    }

    template<class T, class S>
    void memcpy_from_symbol(T *hp, const S& s, size_t N=1)
    {
        cudaMemcpyFromSymbol(hp, s, sizeof(T)*N);
    }

} // end namespace cusp
