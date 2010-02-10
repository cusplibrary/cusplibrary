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

/*! \file diagonal.h
 *  \brief Diagonal preconditioner.
 *  
 *  Contributed by Andrew Trachenko and Nikita Styopin 
 *  at SALD Laboratory ( http://www.saldlab.com )
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/linear_operator.h>

namespace cusp
{
namespace precond
{

template <typename ValueType, typename MemorySpace>
class diagonal : public linear_operator<ValueType, MemorySpace>
{       
    typedef linear_operator<ValueType, MemorySpace> Parent;
    cusp::array1d<ValueType, MemorySpace> diagonal_reciprocals;

public:

    template<typename IndexType2, typename ValueType2, class MemorySpace2>
    diagonal(const cusp::csr_matrix<IndexType2, ValueType2, MemorySpace2>& A);

    template<typename IndexType2, typename ValueType2, class MemorySpace2>
    diagonal(const cusp::coo_matrix<IndexType2, ValueType2, MemorySpace2>& A);

    template<typename IndexType2, typename ValueType2, class MemorySpace2>
    diagonal(const cusp::ell_matrix<IndexType2, ValueType2, MemorySpace2>& A);

    template<typename IndexType2, typename ValueType2, class MemorySpace2>
    diagonal(const cusp::hyb_matrix<IndexType2, ValueType2, MemorySpace2>& A);
        
    template <typename VectorType1, typename VectorType2>
    void operator()(const VectorType1& x, VectorType2& y) const;
};

} // end namespace precond
} // end namespace cusp

#include <cusp/precond/detail/diagonal.inl>

