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

/*! \file
 * \brief 2D array of elements that may reside in "host" or "device" memory space
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/memory.h>
#include <cusp/format.h>
#include <cusp/array1d.h>

#include <cusp/detail/array2d_format_utils.h>
#include <cusp/detail/matrix_base.h>

#include <thrust/functional.h>

namespace cusp
{

// TODO document mapping of (i,j) onto values[pitch * i + j] or values[pitch * j + i]
// TODO document that array2d operations will try to respect .pitch of destination argument

/*! \addtogroup arrays Arrays
 */

/*! \addtogroup array_containers Array Containers
 *  \ingroup arrays
 *  \{
 */

/**
 * \brief The array2d class is a 2D vector container that may contain elements
 * stored in "host" or "device" memory space
 *
 * \tparam T value_type of the array
 * \tparam MemorySpace memory space of the array (cusp::host_memory or cusp::device_memory)
 * \tparam Orientation orientation of the array (cusp::row_major or cusp::column_major)
 *
 * \par Overview
 * A array2d vector is a container that supports random access to elements.
 * The memory associated with a array2d vector may reside in either "host"
 * or "device" memory depending on the supplied allocator embedded in the
 * MemorySpace template argument. array2d vectors represent 2D matrices in
 * either row-major or column-major format.
 *
 * \par Example
 * \code
 * // include cusp array2d header file
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * int main()
 * {
 *   // Allocate a array of size 2 in "host" memory
 *   cusp::array2d<int,cusp::host_memory> a(3,3);
 *
 *   // Set the entries in the matrix using shorthand operator
 *   a(0,0) = 0; a(0,1) = 1; a(0,2) = 2;
 *   a(1,0) = 3; a(1,1) = 4; a(1,2) = 5;
 *   a(2,0) = 6; a(2,1) = 7; a(2,2) = 8;
 *
 *   // Allocate a seceond array2d in "device" memory that is
 *   // a copy of the first but in column major
 *   cusp::array1d<int,cusp::device_memory,cusp::column_major> b(a);
 *
 *   // print row-major layout of data
 *   // [0, 1, 2, 3, 4, 5, 6, 7, 8]
 *   cusp::print(a.values);
 *   // print column-major layout of data
 *   // [0, 3, 6, 1, 4, 7, 2, 5, 8]
 *   cusp::print(b.values);
 * }
 * \endcode
 */
template<typename T, class MemorySpace, class Orientation = cusp::row_major>
class array2d : public cusp::detail::matrix_base<int,T,MemorySpace,cusp::array2d_format>
{
    typedef typename cusp::detail::matrix_base<int,T,MemorySpace,cusp::array2d_format> Parent;

public:
    /*! \cond */
    typedef Orientation orientation;

    template<typename MemorySpace2>
    struct rebind {
        typedef cusp::array2d<T, MemorySpace2, Orientation> type;
    };

    typedef typename cusp::array1d<T, MemorySpace> values_array_type;
    typedef typename cusp::array2d<T, MemorySpace, Orientation> container;

    typedef typename cusp::array2d_view<typename values_array_type::view, Orientation> view;
    typedef typename cusp::array2d_view<typename values_array_type::const_view, Orientation> const_view;

    typedef cusp::detail::row_or_column_view<
      typename values_array_type::iterator,thrust::detail::is_same<Orientation,cusp::row_major>::value>
      row_view_type;

    typedef typename row_view_type::ArrayType row_view;

    typedef cusp::detail::row_or_column_view
      <typename values_array_type::iterator,thrust::detail::is_same<Orientation,cusp::column_major>::value>
      column_view_type;

    typedef typename column_view_type::ArrayType column_view;

    typedef cusp::detail::row_or_column_view
      <typename values_array_type::const_iterator,thrust::detail::is_same<Orientation,cusp::row_major>::value>
      const_row_view_type;

    typedef typename const_row_view_type::ArrayType const_row_view;

    typedef cusp::detail::row_or_column_view
      <typename values_array_type::const_iterator,thrust::detail::is_same<Orientation,cusp::column_major>::value>
      const_column_view_type;

    typedef typename const_column_view_type::ArrayType const_column_view;
    /*! \endcond */

    values_array_type values;

    // minor_dimension + padding
    size_t pitch;

    /*! This constructor creates an empty \p array2d vector.
     */
    array2d()
        : Parent(), pitch(0), values(0) {}

    /*! This constructor creates a array2d vector with the given
     *  shape.
     *  \param num_rows The number of elements to initially create.
     *  \param num_cols The number of elements to initially create.
     */
    array2d(size_t num_rows, size_t num_cols)
        : Parent(num_rows, num_cols, num_rows * num_cols),
          pitch(cusp::detail::minor_dimension(num_rows, num_cols, orientation())),
          values(num_rows * num_cols) {}

    /*! This constructor creates a array2d vector with the given
     *  shape and fills the entries with a given value.
     *  \param num_rows The number of array2d rows.
     *  \param num_cols The number of array2d columns.
     *  \param value The initial value of all entries.
     */
    array2d(size_t num_rows, size_t num_cols, const T& value)
        : Parent(num_rows, num_cols, num_rows * num_cols),
          pitch(cusp::detail::minor_dimension(num_rows, num_cols, orientation())),
          values(num_rows * num_cols, value) {}

    /*! This constructor creates a array2d vector with the given
     *  shape, fills the entries with a given value and sets the pitch
     *  \param num_rows The number of array2d rows.
     *  \param num_cols The number of array2d columns.
     *  \param value The initial value of all entries.
     *  \param pitch The stride between entries in the major dimension.
     */
    array2d(size_t num_rows, size_t num_cols, const T& value, size_t pitch)
        : Parent(num_rows, num_cols, num_rows * num_cols),
          pitch(pitch),
          values(pitch * cusp::detail::major_dimension(num_rows, num_cols, orientation()), value)
    {
        if (pitch < cusp::detail::minor_dimension(num_rows, num_cols, orientation()))
            throw cusp::invalid_input_exception("pitch cannot be less than minor dimension");
    }

    /*! This constructor creates a array2d vector from another matrix
     *  \tparam MatrixType Type of the input matrix
     *  \param matrix Input matrix used to create array2d matrix
     */
    template <typename MatrixType>
    array2d(const MatrixType& matrix);

    /*! Subscript access to the data contained in this array2d.
     *  \param i Row index for which data should be accessed.
     *  \param j Column index for which data should be accessed.
     *  \return Read/write reference to data.
     */
    typename values_array_type::reference operator()(const size_t i, const size_t j)
    {
        return values[cusp::detail::index_of(i, j, pitch, orientation())];
    }

    /*! Subscript access to the data contained in this array2d.
     *  \param i Row index for which data should be accessed.
     *  \param j Column index for which data should be accessed.
     *  \return Read reference to data.
     */
    typename values_array_type::const_reference operator()(const size_t i, const size_t j) const
    {
        return values[cusp::detail::index_of(i, j, pitch, orientation())];
    }

    /*! This method will resize this array2d to the specified number of
     *  dimensions. If the number of total entries is smaller than this
     *  array2d's current size this array2d is truncated, otherwise this
     *  array2d is extended with the value of new entries undefined.
     *
     *  \param num_rows The number of rows this array2d should contain
     *  \param num_cols The number of columns this array2d should contain
     */
    void resize(size_t num_rows, size_t num_cols)
    {
        // preserve .pitch if possible
        if (this->num_rows == num_rows && this->num_cols == num_cols)
            return;

        resize(num_rows, num_cols, cusp::detail::minor_dimension(num_rows, num_cols, orientation()));
    }

    /*! This method will resize this array2d to the specified number of
     *  dimensions. If the number of total entries is smaller than this
     *  array2d's current size this array2d is truncated, otherwise this
     *  array2d is extended with the value of new entries undefined.
     *
     *  \param num_rows The number of rows this array2d should contain
     *  \param num_cols The number of columns this array2d should contain
     *  \param pitch The stride between major dimension entries this array2d
     *  should contain
     */
    void resize(size_t num_rows, size_t num_cols, size_t pitch)
    {
        if (pitch < cusp::detail::minor_dimension(num_rows, num_cols, orientation()))
            throw cusp::invalid_input_exception("pitch cannot be less than minor dimension");

        values.resize(pitch * cusp::detail::major_dimension(num_rows, num_cols, orientation()));

        this->num_rows    = num_rows;
        this->num_cols    = num_cols;
        this->pitch       = pitch;
        this->num_entries = num_rows * num_cols;
    }


    /*! This method swaps the contents of this array2d with another array2d.
     *  \param v The array2d with which to swap.
     */
    void swap(array2d& matrix)
    {
        Parent::swap(matrix);
        thrust::swap(this->pitch, matrix.pitch);
        values.swap(matrix.values);
    }

    /*! Retrieve a raw pointer to the underlying memory contained in the \p
     * array1d vector.
     * \return pointer to first element pointed to by array
     */
    T* raw_data(void)
    {
        return values.raw_data();
    }

    row_view row(size_t i)
    {
        return row_view_type::get_array(*this, i);
    }

    column_view column(size_t i)
    {
        return column_view_type::get_array(*this, i);
    }

    const_row_view row(size_t i) const
    {
        return const_row_view_type::get_array(*this, i);
    }

    const_column_view column(size_t i) const
    {
        return const_column_view_type::get_array(*this, i);
    }

    array2d& operator=(const array2d& matrix);

    template <typename MatrixType>
    array2d& operator=(const MatrixType& matrix);

}; // class array2d
/*! \}
 */

/*! \addtogroup array_views Array Views
 *  \ingroup arrays
 *  \{
 */

/**
 * \brief The array1d_view class is a 1D vector container that wraps data from
 * various iterators in a array1d datatype
 *
 * \tparam Iterator The iterator type used to encapsulate the underlying data.
 *
 * \par Overview
 * array1d_view vector is a container that wraps existing iterators in array1d
 * datatypes to interoperate with cusp algorithms. array1d_view datatypes
 * are interoperable with a wide range of iterators exposed by Thrust and the
 * STL library.
 *
 * \par Example
 * \code
 * // include cusp array1d header file
 * #include <cusp/array1d.h>
 * #include <cusp/print.h>
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
 * \endcode
 */
template<typename ArrayView, class Orientation = cusp::row_major>
class array2d_view
  : public  cusp::detail::matrix_base<int, typename ArrayView::value_type,typename ArrayView::memory_space, cusp::array2d_format>
{
    typedef cusp::detail::matrix_base<int, typename ArrayView::value_type,typename ArrayView::memory_space, cusp::array2d_format> Parent;
public:
    /*! \cond */
    typedef Orientation orientation;

    typedef ArrayView values_array_type;

    typedef typename cusp::array2d<typename Parent::value_type, typename Parent::memory_space, Orientation> container;

    typedef typename cusp::array2d_view<ArrayView, Orientation> view;

    typedef cusp::detail::row_or_column_view<
      typename values_array_type::iterator,thrust::detail::is_same<Orientation,cusp::row_major>::value> row_view_type;
    typedef typename row_view_type::ArrayType row_view;

    typedef cusp::detail::row_or_column_view<
      typename values_array_type::iterator,thrust::detail::is_same<Orientation,cusp::column_major>::value> column_view_type;
    typedef typename column_view_type::ArrayType column_view;
    /*! \endcond */

    values_array_type values;

    // minor_dimension + padding
    size_t pitch;

    // construct empty view
    array2d_view(void)
        : Parent(), values(0), pitch(0) {}

    array2d_view(const array2d_view& a)
        : Parent(a), values(a.values), pitch(a.pitch) {}

    // TODO handle different Orientation (pitch = major)
    //template <typename Array2, typename Orientation2>
    //array2d_view(const array2d_view<Array2,Orientation2>& A)

    // TODO check values.size()

    // construct from array2d container
    array2d_view(array2d<typename Parent::value_type, typename Parent::memory_space, orientation>& a)
        : Parent(a), values(a.values), pitch(a.pitch) {}

    template <typename Array2>
    array2d_view(size_t num_rows, size_t num_cols, size_t pitch, Array2& values)
        : Parent(num_rows, num_cols, num_rows * num_cols), pitch(pitch), values(values) {}

    template <typename Array2>
    array2d_view(size_t num_rows, size_t num_cols, size_t pitch, const Array2& values)
        : Parent(num_rows, num_cols, num_rows * num_cols), pitch(pitch), values(values) {}

    typename values_array_type::reference operator()(const size_t i, const size_t j) const
    {
        return values[cusp::detail::index_of(i, j, pitch, orientation())];
    }

    void resize(size_t num_rows, size_t num_cols, size_t pitch)
    {
        if (pitch < cusp::detail::minor_dimension(num_rows, num_cols, orientation()))
            throw cusp::invalid_input_exception("pitch cannot be less than minor dimension");

        values.resize(pitch * cusp::detail::major_dimension(num_rows, num_cols, orientation()));

        this->num_rows    = num_rows;
        this->num_cols    = num_cols;
        this->pitch       = pitch;
        this->num_entries = num_rows * num_cols;
    }

    void resize(size_t num_rows, size_t num_cols)
    {
        // preserve .pitch if possible
        if (this->num_rows == num_rows && this->num_cols == num_cols)
            return;

        resize(num_rows, num_cols, cusp::detail::minor_dimension(num_rows, num_cols, orientation()));
    }

    row_view row(size_t i)
    {
        return row_view_type::get_array(*this, i);
    }

    column_view column(size_t i)
    {
        return column_view_type::get_array(*this, i);
    }

    row_view row(size_t i) const
    {
        return row_view_type::get_array(*this, i);
    }

    column_view column(size_t i) const
    {
        return column_view_type::get_array(*this, i);
    }
}; // class array2d_view


template <typename Iterator, typename Orientation>
array2d_view<typename cusp::array1d_view<Iterator>,Orientation>
make_array2d_view(size_t num_rows, size_t num_cols, size_t pitch, const cusp::array1d_view<Iterator>& values, Orientation)
{
    return array2d_view<typename cusp::array1d_view<Iterator>,Orientation>(num_rows, num_cols, pitch, values);
}

template <typename Array, typename Orientation>
array2d_view<Array,Orientation>
make_array2d_view(const array2d_view<Array, Orientation>& a)
{
    return array2d_view<Array,Orientation>(a);
}

template<typename T, class MemorySpace, class Orientation>
array2d_view<typename cusp::array1d_view<typename cusp::array1d<T,MemorySpace>::iterator >, Orientation>
make_array2d_view(cusp::array2d<T,MemorySpace,Orientation>& a)
{
    return cusp::make_array2d_view(a.num_rows, a.num_cols, a.pitch, cusp::make_array1d_view(a.values), Orientation());
}

template<typename T, class MemorySpace, class Orientation>
array2d_view<typename cusp::array1d_view<typename cusp::array1d<T,MemorySpace>::const_iterator >, Orientation>
make_array2d_view(const cusp::array2d<T,MemorySpace,Orientation>& a)
{
    return cusp::make_array2d_view(a.num_rows, a.num_cols, a.pitch, cusp::make_array1d_view(a.values), Orientation());
}
/*! \}
 */

} // end namespace cusp

#include <cusp/detail/array2d.inl>

