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

/*! \file array2d.h
 *  \brief Two-dimensional array
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

/*! \p array2d : One-dimensional array container
 *
 * \tparam T value_type of the array
 * \tparam MemorySpace memory space of the array (cusp::host_memory or cusp::device_memory)
 * \tparam Orientation orientation of the array (cusp::row_major or cusp::column_major)
 *
 * \TODO example
 */
template<typename T, class MemorySpace, class Orientation = cusp::row_major>
class array2d : public cusp::detail::matrix_base<int,T,MemorySpace,cusp::array2d_format>
{
    typedef typename cusp::detail::matrix_base<int,T,MemorySpace,cusp::array2d_format> Parent;

public:
    typedef Orientation orientation;

    template<typename MemorySpace2>
    struct rebind {
        typedef cusp::array2d<T, MemorySpace2, Orientation> type;
    };

    typedef typename cusp::array1d<T, MemorySpace> values_array_type;

    /*! equivalent container type
     */
    typedef typename cusp::array2d<T, MemorySpace, Orientation> container;

    /*! equivalent view type
     */
    typedef typename cusp::array2d_view<typename values_array_type::view, Orientation> view;

    /*! equivalent const_view type
     */
    typedef typename cusp::array2d_view<typename values_array_type::const_view, Orientation> const_view;

    /*! array1d_view of a single row
     */
    typedef cusp::detail::row_or_column_view<
      typename values_array_type::iterator,thrust::detail::is_same<Orientation,cusp::row_major>::value>
      row_view_type;

    typedef typename row_view_type::ArrayType row_view;

    /*! array1d_view of a single column
     */
    typedef cusp::detail::row_or_column_view
      <typename values_array_type::iterator,thrust::detail::is_same<Orientation,cusp::column_major>::value>
      column_view_type;

    typedef typename column_view_type::ArrayType column_view;

    /*! const array1d_view of a single row
     */
    typedef cusp::detail::row_or_column_view
      <typename values_array_type::const_iterator,thrust::detail::is_same<Orientation,cusp::row_major>::value>
      const_row_view_type;

    typedef typename const_row_view_type::ArrayType const_row_view;

    /*! const array1d_view of a single column
     */
    typedef cusp::detail::row_or_column_view
      <typename values_array_type::const_iterator,thrust::detail::is_same<Orientation,cusp::column_major>::value>
      const_column_view_type;

    typedef typename const_column_view_type::ArrayType const_column_view;

    values_array_type values;

    // minor_dimension + padding
    size_t pitch;

    // construct empty matrix
    array2d()
        : Parent(), pitch(0), values(0) {}

    // construct matrix with given shape and number of entries
    array2d(size_t num_rows, size_t num_cols)
        : Parent(num_rows, num_cols, num_rows * num_cols),
          pitch(cusp::detail::minor_dimension(num_rows, num_cols, orientation())),
          values(num_rows * num_cols) {}

    // construct matrix with given shape, number of entries and fill with a given value
    array2d(size_t num_rows, size_t num_cols, const T& value)
        : Parent(num_rows, num_cols, num_rows * num_cols),
          pitch(cusp::detail::minor_dimension(num_rows, num_cols, orientation())),
          values(num_rows * num_cols, value) {}

    // construct matrix with given shape, number of entries, pitch and fill with a given value
    array2d(size_t num_rows, size_t num_cols, const T& value, size_t pitch)
        : Parent(num_rows, num_cols, num_rows * num_cols),
          pitch(pitch),
          values(pitch * cusp::detail::major_dimension(num_rows, num_cols, orientation()), value)
    {
        if (pitch < cusp::detail::minor_dimension(num_rows, num_cols, orientation()))
            throw cusp::invalid_input_exception("pitch cannot be less than minor dimension");
    }

    // construct from another matrix
    template <typename MatrixType>
    array2d(const MatrixType& matrix);

    typename values_array_type::reference operator()(const size_t i, const size_t j)
    {
        return values[cusp::detail::index_of(i, j, pitch, orientation())];
    }

    typename values_array_type::const_reference operator()(const size_t i, const size_t j) const
    {
        return values[cusp::detail::index_of(i, j, pitch, orientation())];
    }

    T* raw_data(void)
    {
        return values.raw_data();
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

    void swap(array2d& matrix)
    {
        Parent::swap(matrix);
        thrust::swap(this->pitch, matrix.pitch);
        values.swap(matrix.values);
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

/*! \p array2d_view : One-dimensional array view
 *
 * \tparam Array Underlying one-dimensional array view
 * \tparam Orientation orientation of the array (cusp::row_major or cusp::column_major)
 *
 * \TODO example
 */
template<typename ArrayView, class Orientation = cusp::row_major>
class array2d_view : public cusp::detail::matrix_base<int, typename ArrayView::value_type,typename ArrayView::memory_space, cusp::array2d_format>
{
    typedef cusp::detail::matrix_base<int, typename ArrayView::value_type,typename ArrayView::memory_space, cusp::array2d_format> Parent;
public:
    typedef Orientation orientation;

    typedef ArrayView values_array_type;

    values_array_type values;

    /*! equivalent container type
     */
    typedef typename cusp::array2d<typename Parent::value_type, typename Parent::memory_space, Orientation> container;

    /*! equivalent view type
     */
    typedef typename cusp::array2d_view<ArrayView, Orientation> view;

    /*! array1d_view of a single row
     */
    typedef cusp::detail::row_or_column_view<typename values_array_type::iterator,thrust::detail::is_same<Orientation,cusp::row_major>::value> row_view_type;
    typedef typename row_view_type::ArrayType row_view;

    /*! array1d_view of a single column
     */
    typedef cusp::detail::row_or_column_view<typename values_array_type::iterator,thrust::detail::is_same<Orientation,cusp::column_major>::value> column_view_type;
    typedef typename column_view_type::ArrayType column_view;

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

