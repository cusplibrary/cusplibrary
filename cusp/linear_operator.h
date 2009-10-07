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

#include <cusp/detail/host/spmv.h>
#include <cusp/detail/device/spmv.h>

namespace cusp
{

template <class MatrixType, class SpMV>
struct linear_operator : public MatrixType
{
    typedef typename MatrixType::value_type value_type;
    typedef typename MatrixType::memory_space memory_space;

    SpMV spmv;

    linear_operator(MatrixType& _matrix, SpMV _spmv) : MatrixType(_matrix), spmv(_spmv) {}

    template <typename ValueType>
    void operator()(const ValueType * x, ValueType * y) { spmv(*this, x, y); }

}; // end linear_operator


template <class MatrixType>
struct default_linear_operator : public MatrixType
{
    typedef typename MatrixType::value_type value_type;
    typedef typename MatrixType::memory_space memory_space;
    
    default_linear_operator(MatrixType& _matrix) : MatrixType(_matrix) {}

    template <typename ValueType>
    void spmv(const ValueType * x, ValueType * y, cusp::host_memory){
        cusp::detail::host::spmv(*this, x, y);
    }
    
    template <typename ValueType>
    void spmv(const ValueType * x, ValueType * y, cusp::device_memory){
        cusp::detail::device::spmv(*this, x, y);
    }

    template <typename ValueType>
    void operator()(const ValueType * x, ValueType * y) { 
        typedef typename MatrixType::memory_space MemorySpace;
        spmv(x, y, MemorySpace()); 
    }
}; // end default_linear_operator




template <class MatrixType, class SpMV>
linear_operator<MatrixType, SpMV> make_linear_operator(MatrixType& matrix, SpMV spmv)
{
    return linear_operator<MatrixType, SpMV>(matrix, spmv);
}
   
template <class MatrixType>
cusp::default_linear_operator<MatrixType> make_linear_operator(MatrixType& matrix)
{
    return default_linear_operator<MatrixType>(matrix);
}

template <class MatrixType, class SpMV>
linear_operator<MatrixType, SpMV> make_linear_operator(linear_operator<MatrixType,SpMV>& A)
{
    return A;
}

} // end namespace cusp

