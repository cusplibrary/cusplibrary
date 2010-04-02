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

#include <cusp/detail/convert.h>

namespace cusp
{

//////////////////
// Constructors //
//////////////////

// construct empty matrix
template<typename ValueType, class MemorySpace, class Orientation>
array2d<ValueType,MemorySpace,Orientation>
    ::array2d() {}

// construct matrix with given shape and number of entries
template<typename ValueType, class MemorySpace, class Orientation>
array2d<ValueType,MemorySpace,Orientation>
    ::array2d(int num_rows, int num_cols)
        : detail::matrix_base<int,ValueType,MemorySpace>(num_rows, num_cols, num_rows * num_cols),
          values(num_rows * num_cols) {}

// construct matrix with given shape and number of entries and fill with a given value
template<typename ValueType, class MemorySpace, class Orientation>
array2d<ValueType,MemorySpace,Orientation>
    ::array2d(int num_rows, int num_cols, const ValueType& value)
        : detail::matrix_base<int,ValueType,MemorySpace>(num_rows, num_cols, num_rows * num_cols),
          values(num_rows * num_cols, value) {}

// construct from another array2d
template<typename ValueType, class MemorySpace, class Orientation>
template <typename ValueType2, typename MemorySpace2>
array2d<ValueType,MemorySpace,Orientation>
    ::array2d(const array2d<ValueType2, MemorySpace2, Orientation>& matrix)
        : detail::matrix_base<int,ValueType,MemorySpace>(matrix.num_rows, matrix.num_cols, matrix.num_entries),
          values(matrix.values) {}

// construct from a different matrix format
template<typename ValueType, class MemorySpace, class Orientation>
template <typename MatrixType>
array2d<ValueType,MemorySpace,Orientation>
    ::array2d(const MatrixType& matrix)
    {
        cusp::detail::convert(*this, matrix);
    }

//////////////////////
// Member Functions //
//////////////////////

template<typename ValueType, class MemorySpace, class Orientation>
    void
    array2d<ValueType,MemorySpace,Orientation>
    ::resize(int num_rows, int num_cols)
{
    this->num_rows    = num_rows;
    this->num_cols    = num_cols;
    this->num_entries = num_rows * num_cols;

    values.resize(num_rows * num_cols);
}

template<typename ValueType, class MemorySpace, class Orientation>
    void
    array2d<ValueType,MemorySpace,Orientation>
    ::swap(array2d<ValueType,MemorySpace,Orientation> & matrix)
{
    detail::matrix_base<int,ValueType,MemorySpace>::swap(matrix);

    values.swap(matrix.values);
}

template <typename ValueType, class MemorySpace, class Orientation>
template <typename ValueType2, typename MemorySpace2>
    array2d<ValueType,MemorySpace,Orientation>&
    array2d<ValueType,MemorySpace,Orientation>
    ::operator=(const array2d<ValueType2, MemorySpace2, Orientation>& matrix)
    {
        this->num_rows    = matrix.num_rows;
        this->num_cols    = matrix.num_cols;
        this->num_entries = matrix.num_entries;
        this->values      = matrix.values;

        return *this;
    }

template <typename ValueType, class MemorySpace, class Orientation>
template <typename MatrixType>
    array2d<ValueType,MemorySpace,Orientation>&
    array2d<ValueType,MemorySpace,Orientation>
    ::operator=(const MatrixType& matrix)
    {
        cusp::detail::convert(*this, matrix);
        
        return *this;
    }

template<typename ValueType1, typename MemorySpace1, typename Orientation1,
         typename ValueType2, typename MemorySpace2>
bool operator==(const array2d<ValueType1,MemorySpace1,Orientation1>& lhs,
                const array2d<ValueType2,MemorySpace2,Orientation1>& rhs)
{
    // TODO generalize to mixed orientations
    if (lhs.num_rows != rhs.num_rows || lhs.num_cols != rhs.num_cols)
        return false;

    return lhs.values == rhs.values;
}

template<typename ValueType1, typename MemorySpace1, typename Orientation1,
         typename ValueType2, typename MemorySpace2>
bool operator!=(const array2d<ValueType1,MemorySpace1,Orientation1>& lhs,
                const array2d<ValueType2,MemorySpace2,Orientation1>& rhs)
{
    return !(lhs == rhs);
}
    
} // end namespace cusp

