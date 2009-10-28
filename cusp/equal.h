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

#include <cusp/array2d.h>

namespace cusp
{

template <typename ValueType1, class Orientation1, typename ValueType2,  class Orientation2>
bool equal(const cusp::array2d<ValueType1, cusp::host_memory, Orientation1>& A,
           const cusp::array2d<ValueType2, cusp::host_memory, Orientation2>& B)
{
    if (A.num_rows != B.num_rows || A.num_cols != B.num_cols)
        return false;

    for(size_t i = 0; i < A.num_rows; i++)
        for(size_t j = 0; j < B.num_rows; j++)
            if(A(i,j) != B(i,j))
                return false;

    return true;
}

} // end namespace cusp

