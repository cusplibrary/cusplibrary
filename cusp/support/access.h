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
    template <typename I, typename T>
    T get_device_array_element(const T * d_ptr, const I i)
    {
        T h_element;
        memcpy_to_host(&h_element, d_ptr + i, 1);
        return h_element;
    }
    template <typename I, typename T>
    void set_device_array_element(T * d_ptr, const I i, const T& h_element)
    {
        memcpy_to_device(d_ptr + i, &h_element, 1);
    }
   
    // host path
    template <typename I, typename T>
    T get_array_element(const T * ptr, const I i, cusp::host_memory)
    { 
        return ptr[i];
    }
    
    template <typename I, typename T>
    void set_array_element(T * ptr, const I i, const T& element, cusp::host_memory)
    {
        ptr[i] = element;
    }
   
    // specialize for device
    template <typename I, typename T>
    T get_array_element(const T * ptr, const I i, cusp::device_memory)
    { 
        return get_device_array_element(ptr, i);
    }
    
    template <typename I, typename T>
    void set_array_element(T * ptr, const I i, const T& element, cusp::device_memory)
    {
        set_device_array_element(ptr, i, element);
    }
    
    template <class MemorySpace, typename I, typename T>
    T get_array_element(const T * ptr, const I i)
    { 
        return get_array_element(ptr, i, MemorySpace());
    }
    
    template <class MemorySpace, typename I, typename T>
    void set_array_element(T * ptr, const I i, const T& element)
    {
        set_array_element(ptr, i, element, MemorySpace());
    }
 
} // end namespace cusp
