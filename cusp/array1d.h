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

/*! \file array1d.h
 *  \brief One-dimensional array of elements that may reside in "host" or
 *  "device" memory space
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/memory.h>
#include <cusp/format.h>
#include <cusp/exception.h>

#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/detail/vector_base.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

template <typename T>
struct has_iterator
{
    template <typename U>
    static char helper(typename U::iterator* x);

    template <typename U>
    static long helper(U* x);

    static const bool value = (sizeof(helper<T>(0)) == 1);
};

namespace cusp
{

// forward declaration of array1d_view
template <typename RandomAccessIterator> class array1d_view;

/*! \addtogroup arrays Arrays
 */

/*! \addtogroup array_containers Array Containers
 *  \ingroup arrays
 *  \{
 */

/*! A \p array1d vector is a container that supports random access to elements.
 * The memory associated with a \p array1d vector may reside in either "host"
 * or "device" memory depending on the supplied allocator embedded in the
 * MemorySpace template argument. \p array1d vectors inherit a considerable
 * amount of functionality from the thrust::detail::vector_base case.
 *
 * \tparam T value_type of the array
 * \tparam MemorySpace memory space of the array (cusp::host_memory or cusp::device_memory)
 *
 * \code
 * #include <cusp/array1d.h>  // Include cusp array1d header file
 *
 * int main()
 * {
 *   // Allocate a array of size 2 in "host" memory
 *   cusp::array1d<int,cusp::host_memory> a(2);
 *
 *   // Set the first element to 0 and second element to 1
 *   a[0] = 0;
 *   a[1] = 1;
 *
 *   // Allocate a seceond array in "device" memory that is
 *   // a copy of the first
 *   cusp::array1d<int,cusp::device_memory> b(a);
 * }
 * \endcode
 */
template <typename T, typename MemorySpace>
class array1d : public thrust::detail::vector_base<T, typename cusp::default_memory_allocator<T, MemorySpace>::type>
{
private:
    typedef typename cusp::default_memory_allocator<T, MemorySpace>::type Alloc;
    typedef typename thrust::detail::vector_base<T,Alloc> Parent;

public:
    /*! \cond */
    typedef MemorySpace memory_space;
    typedef cusp::array1d_format format;
    /*! \endcond */

    /*! equivalent container type
     */
    typedef typename cusp::array1d<T,MemorySpace> container;

    /*! equivalent container type in another MemorySpace
     */
    template<typename MemorySpace2>
    struct rebind {
        typedef cusp::array1d<T, MemorySpace2> type;
    };

    /*! equivalent view type
     */
    typedef typename cusp::array1d_view<typename Parent::iterator> view;

    /*! equivalent const_view type
     */
    typedef typename cusp::array1d_view<typename Parent::const_iterator> const_view;

    /*! associated size_type
     */
    typedef typename Parent::size_type  size_type;

    /*! associated value_type
     */
    typedef typename Parent::value_type value_type;

    /*! This constructor creates an empty \p array1d vector.
     */
    array1d(void) : Parent() {}

    /*! This constructor creates a \p array1d vector with the given
     *  size.
     *  \param n The number of elements to initially create.
     */
    explicit array1d(size_type n)
        : Parent()
    {
        if(n > 0)
        {
            Parent::m_storage.allocate(n);
            Parent::m_size = n;
        }
    }

    /*! This constructor creates a \p array1d vector with copies
     *  of an exemplar element.
     *  \param n The number of elements to initially create.
     *  \param value An element to copy.
     */
    explicit array1d(size_type n, const value_type &value)
        : Parent(n, value) {}

    /*! Copy constructor copies from an exemplar array with iterator
     *  \tparam Array Input array type supporting iterators
     *  \param v The vector to copy.
     */
    template<typename ArrayType>
    array1d(const ArrayType &v,
            typename thrust::detail::enable_if<has_iterator<ArrayType>::value>::type* = 0)
        : Parent(v.begin(), v.end()) {}

    /*! This constructor builds a \p array1d vector from a range.
     *  \tparam Iterator iterator type of \p array1d_view
     *  \param first The beginning of the range.
     *  \param last The end of the range.
     */
    template<typename Iterator>
    array1d(Iterator first, Iterator last)
        : Parent(first, last) {}

    /*! Assign operator copies from an exemplar \p array1d vector.
     *  \param v The \p array1d vector to copy.
     */
    template<typename ArrayType>
    array1d &operator=(const ArrayType& a)
    {
        Parent::assign(a.begin(), a.end());
        return *this;
    }

    /*! Extract a small vector from a \p array1d vector.
     *  \param start_index The starting index of the sub-array.
     *  \param num_entries The number of entries in the sub-array.
     */
    view subarray(size_type start_index, size_type num_entries)
    {
        return view(Parent::begin() + start_index, Parent::begin() + start_index + num_entries);
    }

    /*! Retrieve a raw pointer to the underlying memory contained in the \p
     * array1d vector.
     */
    T* raw_data(void)
    {
        return thrust::raw_pointer_cast(&Parent::m_storage[0]);
    }

    /*! Retrieve a raw const pointer to the underlying memory contained in
     * the \p array1d vector.
     */
    const T* raw_data(void) const
    {
        return thrust::raw_pointer_cast(&Parent::m_storage[0]);
    }
}; // class array1d
/*! \}
 */

/*! \addtogroup array_views Array Views
 *  \ingroup arrays
 *  \{
 */

/*! A \p array1d_view vector is a container that wraps existing iterators in \p array1d
 * datatypes to interoperate with cusp algorithms. \p array1d_view datatypes
 * are interoperable with a wide range of iterators exposed by Thrust and the
 * STL library.
 *
 * \parame Iterator The iterator type used to encapsulate the underlying data.
 * #include <cusp/array1d.h>  // Include cusp array1d header file
 *
 * int main()
 * {
 *   // Define the container type
 *   typedef cusp::array1d<int, cusp::device_memory Array;
 *
 *   // Get reference to array view type
 *   typedef Array::view View;
 *
 *   // Allocate array1d container with 10 elements
 *   Array array(10,0);
 *
 *   // Create view to the first 5 elements of the array
 *   View first_half(array.begin(), array.begin() + 5);
 *
 *   // Update entries in first_half
 *   first_half[0] = 0; first_half[1] = 1; first_half[2] = 2;
 *
 *   // print the array with updated values
 *   cusp::print(array);
 * }
 */
template<typename Iterator>
class array1d_view : public thrust::iterator_adaptor<array1d_view<Iterator>, Iterator>
{
public :

    typedef cusp::array1d_format format;
    typedef Iterator iterator;

    typedef thrust::iterator_adaptor<array1d_view<iterator>, iterator>  super_t;
    typedef typename cusp::array1d_view<iterator>                       view;

    typedef typename super_t::value_type                                value_type;
    typedef typename super_t::pointer                                   pointer;
    typedef typename super_t::reference                                 reference;
    typedef typename super_t::difference_type                           size_type;
    typedef typename super_t::difference_type                           difference_type;
    typedef typename thrust::iterator_system<iterator>::type            memory_space;

    array1d_view(void)
        : m_size(0), m_capacity(0) {}

    template <typename Array>
    array1d_view(Array& a)
        : super_t(a.begin()), m_size(a.size()), m_capacity(a.capacity()) {}

    template <typename InputIterator>
    array1d_view(InputIterator begin, InputIterator end)
        : super_t(begin), m_size(end-begin), m_capacity(end-begin) {}

    friend class thrust::iterator_core_access;

    reference front(void) const
    {
        return *begin();
    }

    reference back(void) const
    {
        return *(begin() + (size() - 1));
    }

    reference operator[](difference_type n) const
    {
        return *(begin() + n);
    }

    iterator begin(void) const
    {
        return this->base();
    }

    iterator end(void) const
    {
        return begin() + m_size;
    }

    size_type size(void) const
    {
        return m_size;
    }

    size_type capacity(void) const
    {
        return m_capacity;
    }

    pointer data(void)
    {
        return &front();
    }

    const pointer data(void) const
    {
        return &front();
    }

    value_type* raw_data(void)
    {
        return thrust::raw_pointer_cast(data());
    }

    const value_type* raw_data(void) const
    {
        return thrust::raw_pointer_cast(data());
    }

    // TODO : Check if iterator is trivial
    // typename thrust::detail::enable_if< thrust::detail::is_trivial_iterator<iterator>::value, value_type* >::type
    // raw_data(void)
    // {
    //     return thrust::raw_pointer_cast(&front());
    // }

    void resize(size_type new_size)
    {
        if (new_size <= m_capacity)
            m_size = new_size;
        else
            throw cusp::not_implemented_exception("array1d_view cannot resize() larger than capacity()");
    }

    view subarray(size_type start_index, size_type num_entries)
    {
        return view(begin() + start_index, begin() + start_index + num_entries);
    }

protected:
    size_type m_size;
    size_type m_capacity;

private :
};

/*! \p counting_array : One-dimensional counting array view
 *
 * \tparam ValueType iterator type
 *
 * \TODO example
 */
template <typename ValueType>
class counting_array : public cusp::array1d_view< thrust::counting_iterator<ValueType> >
{
private:

    typedef cusp::array1d_view< thrust::counting_iterator<ValueType> > Parent;

public:

    typedef typename Parent::iterator iterator;

    counting_array(ValueType size) : Parent(iterator(0), iterator(size)) {}
    counting_array(ValueType start, ValueType finish) : Parent(iterator(start), iterator(finish)) {}
};

/*! \p constant_array : One-dimensional constant array view
 *
 * \tparam ValueType iterator type
 *
 * \TODO example
 */
template <typename ValueType>
class constant_array : public cusp::array1d_view< thrust::constant_iterator<ValueType> >
{
private:

    typedef cusp::array1d_view< thrust::constant_iterator<ValueType> > Parent;

public:

    typedef typename Parent::iterator iterator;

    constant_array(ValueType value, size_t size) : Parent(iterator(value), iterator(value) + size) {}
};
/*! \}
 */

/* Convenience functions */

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
/*! \}
 */

} // end namespace cusp

#include <cusp/detail/array1d.inl>

