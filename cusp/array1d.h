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

#include <cusp/memory.h>
#include <cusp/format.h>
#include <cusp/exception.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/detail/vector_base.h>

namespace cusp
{
  // forward definitions
  template <typename RandomAccessIterator> class array1d_view;

  template <typename T, typename MemorySpace>
  class array1d : public thrust::detail::vector_base<T, typename cusp::default_memory_allocator<T, MemorySpace>::type>
  {
      private:
          typedef typename cusp::default_memory_allocator<T, MemorySpace>::type Alloc;
          typedef typename thrust::detail::vector_base<T,Alloc> Parent;
  
      public:
          typedef MemorySpace memory_space;
          typedef cusp::array1d_format format;

          /*! equivalent container type
           */
          typedef typename cusp::array1d<T,MemorySpace> container;
          
          /*! equivalent view type
           */
          typedef typename cusp::array1d_view<typename Parent::iterator> view;
          
          /*! equivalent const_view type
           */
          typedef typename cusp::array1d_view<typename Parent::const_iterator> const_view;
  
          typedef typename Parent::size_type  size_type;
          typedef typename Parent::value_type value_type;
  
          array1d(void) : Parent() {}

          explicit array1d(size_type n)
              : Parent()
          {
              if(n > 0)
              {
#if (THRUST_VERSION < 100300)
                  Parent::mBegin = Parent::mAllocator.allocate(n);
                  Parent::mSize  = Parent::mCapacity = n;
#else                    
                  Parent::m_storage.allocate(n);
                  Parent::m_size = n;
#endif
              }
          }
          
          array1d(size_type n, const value_type &value) 
            : Parent(n, value) {}

          template<typename Array>
            array1d(const Array& a, typename thrust::detail::enable_if<!thrust::detail::is_convertible<Array,size_type>::value>::type * = 0)
            : Parent(a.begin(), a.end()) {}

          template<typename InputIterator>
            array1d(InputIterator first, InputIterator last)
            : Parent(first, last) {}

          template<typename Array>
            array1d &operator=(const Array& a)
            { Parent::assign(a.begin(), a.end()); return *this; }

          // TODO specialize resize()
  };
  
  template <typename RandomAccessIterator>
  class array1d_view
  {
    public:
      // what about const_iterator and const_reference?
      typedef RandomAccessIterator                                             iterator;
      typedef cusp::array1d_format                                             format;
      typedef typename thrust::iterator_reference<RandomAccessIterator>::type  reference;
      typedef typename thrust::iterator_difference<RandomAccessIterator>::type difference_type;
      typedef typename thrust::iterator_value<RandomAccessIterator>::type      value_type;
      typedef typename thrust::iterator_space<RandomAccessIterator>::type      memory_space;
          
      /*! equivalent container type
       */
      typedef typename cusp::array1d<value_type,memory_space> container;
      
      /*! equivalent view type
       */
      typedef typename cusp::array1d_view<RandomAccessIterator> view;
  
      // is this right?
      typedef size_t size_type;
      
      array1d_view(void)
        : m_begin(), m_size(0), m_capacity(0) {}
  
      template <typename Array>
      array1d_view(Array& a)
        : m_begin(a.begin()), m_size(a.size()), m_capacity(a.capacity()) {}
      
      template <typename Array>
      array1d_view(const Array& a)
        : m_begin(a.begin()), m_size(a.size()), m_capacity(a.capacity()) {}
 
      // should these be templated?
      array1d_view(RandomAccessIterator first, RandomAccessIterator last)
        : m_begin(first), m_size(last - first), m_capacity(last - first) {}
      
      template <typename Array>
      array1d_view &operator=(Array &a)
      {
        m_begin    = a.begin();
        m_size     = a.size();
        m_capacity = a.capacity();
        return *this;
      }
 
      reference operator[](difference_type n) const
      {
        return m_begin[n];
      }
  
      iterator begin(void) const
      {
        return m_begin;
      }
  
      iterator end(void) const
      {
        return m_begin + m_size;
      }
  
      size_type size(void) const
      {
        return m_size;
      }
  
      size_type capacity(void) const
      {
        return m_capacity;
      }
  
      // TODO is there any value in supporting the two-argument form?
      //      i.e.  void resize(size_type new_size, value_type x = value_type())
      void resize(size_type new_size)
      {
        if (new_size <= m_capacity)
          m_size = new_size;
        else
          // XXX is not_implemented_exception the right choice?
          throw cusp::not_implemented_exception("array1d_view cannot resize() larger than capacity()");
      }

    protected:
      iterator  m_begin;
      size_type m_size;
      size_type m_capacity;
  };
  
  template <typename Iterator>
  array1d_view<Iterator> make_array1d_view(Iterator first, Iterator last)
  {
    return array1d_view<Iterator>(first, last);
  }

  template <typename Iterator>
  array1d_view<Iterator> make_array1d_view(const array1d_view<Iterator>& a)
  {
    return make_array1d_view(a.begin(), a.end());
  }
  
  template <typename T, typename MemorySpace>
  typename array1d<T,MemorySpace>::view make_array1d_view(array1d<T,MemorySpace>& a)
  {
    return make_array1d_view(a.begin(), a.end());
  }
  
  template <typename T, typename MemorySpace>
  typename array1d<T,MemorySpace>::const_view make_array1d_view(const array1d<T,MemorySpace>& a)
  {
    return make_array1d_view(a.begin(), a.end());
  }
  
} // end namespace cusp

#include <cusp/detail/array1d.inl>

