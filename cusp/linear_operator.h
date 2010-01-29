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

#include <cusp/blas.h>
#include <cusp/exception.h>

namespace cusp
{

template <typename ValueType, typename MemorySpace>
class linear_operator
{
    public:
    typedef ValueType   value_type;
    typedef MemorySpace memory_space;

    size_t num_rows;
    size_t num_cols;
    size_t num_entries;

    linear_operator() 
        : num_rows(0), num_cols(0), num_entries(0) {}
    
    linear_operator(size_t num_rows, size_t num_cols)
        : num_rows(num_rows), num_cols(num_cols), num_entries(0) {}

    linear_operator(size_t num_rows, size_t num_cols, size_t num_entries)
        : num_rows(num_rows), num_cols(num_cols), num_entries(num_entries) {}

    template <typename VectorType1, typename VectorType2>
    void operator()(const VectorType1& x, VectorType2& y) const
    {
        throw cusp::not_implemented_exception("linear_operator does not implement y <- A*x");
    }
}; // linear_operator

template <typename ValueType, typename MemorySpace>
class identity_operator : public linear_operator<ValueType,MemorySpace>
{       
    typedef linear_operator<ValueType,MemorySpace> Parent;
    public:

    identity_operator() 
        : Parent() {}
    
    identity_operator(size_t num_rows, size_t num_cols)
        : Parent(num_rows, num_cols) {}

    identity_operator(size_t num_rows, size_t num_cols, size_t num_entries)
        : Parent(num_rows, num_cols, num_entries) {}
    
    template <typename VectorType1,
              typename VectorType2>
    void operator()(const VectorType1& x, VectorType2& y) const
    {
        cusp::blas::copy(x, y);
    }
}; // identity_operator

} // end namespace cusp

