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

#include <cusp/detail/host/elementwise_operation.h>
#include <cusp/detail/device/elementwise_operation.h>

#include <thrust/functional.h>

namespace cusp
{

namespace dispatch
{

template <class MatrixType, class BinaryOperator>
void elementwise_operation(MatrixType& C, const MatrixType& A, const MatrixType& B, BinaryOperator op,
                           cusp::host)
{
    cusp::detail::host::elementwise_operation(C, A, B, op);
}

template <class MatrixType, class BinaryOperator>
void elementwise_operation(MatrixType& C, const MatrixType& A, const MatrixType& B, BinaryOperator op,
                           cusp::device)
{
    cusp::detail::device::elementwise_operation(C, A, B, op);
}

} // end namespace dispatch


template <class MatrixType, class BinaryOperator>
void elementwise_operation(MatrixType& C, const MatrixType& A, const MatrixType& B, BinaryOperator op)
{
    typedef typename MatrixType::memory_space MemorySpace;
    cusp::dispatch::elementwise_operation(C, A, B, op, MemorySpace());
}

    
template <class MatrixType>
void add(MatrixType& C, const MatrixType& A, const MatrixType& B)
{
    typedef typename MatrixType::value_type ValueType;
    cusp::elementwise_operation(C, A, B, thrust::plus<ValueType>());
}


} // end namespace cusp

