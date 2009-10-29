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

#include <cusp/convert.h>

namespace cusp
{

//////////////////
// Constructors //
//////////////////

// construct empty matrix
template<typename ValueType, class SpaceOrAlloc, class Orientation>
array2d<ValueType,SpaceOrAlloc,Orientation>
    ::array2d()
        : matrix_shape<index_type>(0,0),
          num_entries(0) {}

// construct matrix with given shape and number of entries
template<typename ValueType, class SpaceOrAlloc, class Orientation>
array2d<ValueType,SpaceOrAlloc,Orientation>
    ::array2d(size_t num_rows, size_t num_cols)
        : matrix_shape<index_type>(num_rows, num_cols),
          num_entries(num_rows * num_cols),
          values(num_rows * num_cols) {}

// construct from another array2d
template<typename ValueType, class SpaceOrAlloc, class Orientation>
template <typename ValueType2, typename SpaceOrAlloc2>
array2d<ValueType,SpaceOrAlloc,Orientation>
    ::array2d(const array2d<ValueType2, SpaceOrAlloc2, Orientation>& matrix)
        : matrix_shape<index_type>(matrix),
          num_entries(matrix.num_entries),
          values(matrix.values) {}

// construct from a different matrix format
template<typename ValueType, class SpaceOrAlloc, class Orientation>
template <typename MatrixType>
array2d<ValueType,SpaceOrAlloc,Orientation>
    ::array2d(const MatrixType& matrix)
    {
        cusp::convert(*this, matrix);
    }

//////////////////////
// Member Functions //
//////////////////////

template<typename ValueType, class SpaceOrAlloc, class Orientation>
    void
    array2d<ValueType,SpaceOrAlloc,Orientation>
    ::resize(index_type num_rows, index_type num_cols)
{
    values.resize(num_rows * num_cols);

    this->num_rows    = num_rows;
    this->num_cols    = num_cols;
    this->num_entries = num_rows * num_cols;
}

template<typename ValueType, class SpaceOrAlloc, class Orientation>
    void
    array2d<ValueType,SpaceOrAlloc,Orientation>
    ::swap(array2d<ValueType,SpaceOrAlloc,Orientation> & matrix)
{
    values.swap(matrix.values);

    thrust::swap(num_entries, matrix.num_entries);
    
    matrix_shape<index_type>::swap(matrix);
}

template <typename ValueType, class SpaceOrAlloc, class Orientation>
template <typename ValueType2, typename SpaceOrAlloc2>
    array2d<ValueType,SpaceOrAlloc,Orientation>&
    array2d<ValueType,SpaceOrAlloc,Orientation>
    ::operator=(const array2d<ValueType2, SpaceOrAlloc2, Orientation>& matrix)
    {
        this->values           = matrix.values;
        this->num_entries      = matrix.num_entries;
        this->num_rows         = matrix.num_rows;
        this->num_cols         = matrix.num_cols;

        return *this;
    }

template <typename ValueType, class SpaceOrAlloc, class Orientation>
template <typename MatrixType>
    array2d<ValueType,SpaceOrAlloc,Orientation>&
    array2d<ValueType,SpaceOrAlloc,Orientation>
    ::operator=(const MatrixType& matrix)
    {
        cusp::convert(*this, matrix);
        
        return *this;
    }

} // end namespace cusp

