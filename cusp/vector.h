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

#include <thrust/detail/vector_base.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>

#include <thrust/device_allocator.h>

#include <thrust/iterator/iterator_traits.h>

namespace cusp
{
    using thrust::host_vector;
    using thrust::device_vector;

    typedef thrust::device_space_tag device;
    typedef thrust::host_space_tag   host;

    
    template<typename T, typename Space> 
    struct standard_memory_allocator
    {
        typedef Space type; // ASSUME Space is an allocator
    };        

    template<typename T>
    struct standard_memory_allocator<T, cusp::device>
    {
        typedef thrust::device_malloc_allocator<T> type;
    };

    template<typename T>
    struct standard_memory_allocator<T, cusp::host>
    {
        typedef std::allocator<T> type;
    };

    template <typename Alloc>
    struct allocator_space
    {
        typedef typename thrust::iterator_space<typename Alloc::pointer>::type type;
    };



    template <typename T, typename Space>
    class vector : public thrust::detail::vector_base<T, typename standard_memory_allocator<T, Space>::type>
    {
        private:
            typedef typename standard_memory_allocator<T, Space>::type Alloc;
            typedef typename thrust::detail::vector_base<T,Alloc> Parent;

        public:
            typedef typename Parent::size_type  size_type;
            typedef typename Parent::value_type value_type;

            vector(void) : Parent() {}
            
            explicit vector(size_type n)
                : Parent()
            {
                if(n > 0)
                {
                    Parent::mBegin = Parent::mAllocator.allocate(n);
                    Parent::mSize  = Parent::mCapacity = n;
                }
            }

            vector(size_type n, const value_type &value) 
                : Parent(n, value) {}
    
            vector(const vector &v)
                : Parent(v) {}

            template<typename OtherT, typename OtherAlloc>
                vector(const vector<OtherT,OtherAlloc> &v)
                : Parent(v) {}

            template<typename OtherT, typename OtherAlloc>
                vector &operator=(const vector<OtherT,OtherAlloc> &v)
                { Parent::operator=(v); return *this; }

            template<typename OtherT, typename OtherAlloc>
                vector(const std::vector<OtherT,OtherAlloc> &v)
                : Parent(v) {}

            template<typename OtherT, typename OtherAlloc>
                vector &operator=(const std::vector<OtherT,OtherAlloc> &v)
                { Parent::operator=(v); return *this;}

            template<typename InputIterator>
                vector(InputIterator first, InputIterator last)
                : Parent(first, last) {}
    };

} // end namespace cusp

