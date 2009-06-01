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
#include <stdexcept>

namespace cusp
{
    ///////////////////
    // Memory Spaces //
    ///////////////////
    
    struct host_memory
    {
        template<typename T>
        T *allocate(size_t size) { return new T[size]; }

        template<typename T>
        void deallocate(T * p) { delete[] p; }
    };
    
    struct device_memory
    {
        template<typename T>
        T *allocate(size_t size)
        {
            void *ptr;
            cudaError_t rc = cudaMalloc(&ptr, sizeof(T)*size);

            if (rc == cudaSuccess)
                return static_cast<T*>(ptr);
            else
                throw std::bad_alloc();
        }

        template<typename T>
        void deallocate(T * p) { cudaFree(p); }
    };

    struct pinned_memory
    {
        template<typename T>
        T *allocate(size_t size)
        {
            void *ptr;
            cudaError_t rc = cudaMallocHost(&ptr, sizeof(T)*size);
            
            if (rc == cudaSuccess)
                return static_cast<T*>(ptr);
            else
                throw std::bad_alloc();
        }

        template<typename T>
        void deallocate(T * p) { cudaFreeHost(p); }
    };

    ////////////////////
    // New and Delete //
    ////////////////////

    template<typename T, class MemorySpace>
    T * new_array(size_t size, MemorySpace ms = MemorySpace())
    {
        return ms.template allocate<T>(size);
    }

    template<typename T, class MemorySpace>
    void delete_array(T * ptr, MemorySpace ms = MemorySpace())
    {
        ms.deallocate(ptr);
    }

    
    template<typename T>
    T * new_host_array(size_t size)
    {
        return new_array<T>(size, host_memory());
    }

    template<typename T>
    void delete_host_array(T * ptr)
    {
        delete_array(ptr, host_memory());
    }


    template<typename T>
    T * new_device_array(size_t size)
    {
        return new_array<T>(size, device_memory());
    }

    template<typename T>
    void delete_device_array(T * ptr)
    {
        delete_array(ptr, device_memory());
    }

} // end namespace cusp
