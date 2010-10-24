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

// construct from a different matrix
template<typename ValueType, class MemorySpace, class Orientation>
template <typename MatrixType>
array2d<ValueType,MemorySpace,Orientation>
    ::array2d(const MatrixType& matrix)
    {
        cusp::convert(matrix, *this);
    }

//////////////////////
// Member Functions //
//////////////////////

template <typename ValueType, class MemorySpace, class Orientation>
template <typename MatrixType>
    array2d<ValueType,MemorySpace,Orientation>&
    array2d<ValueType,MemorySpace,Orientation>
    ::operator=(const MatrixType& matrix)
    {
        cusp::convert(matrix, *this);
        
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

