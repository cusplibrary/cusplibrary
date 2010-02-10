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

#include <cusp/coo_matrix.h>
#include <cusp/array2d.h>

#include <cusp/detail/generic/multiply.h>
#include <cusp/detail/device/spmv.h>

namespace cusp
{
namespace detail
{
namespace device
{

//////////////////////////////////
// Matrix-Matrix Multiplication //
//////////////////////////////////
template <typename IndexType,
          typename ValueType,
          typename MemorySpace>
void multiply(const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& A,
              const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& B,
                    cusp::coo_matrix<IndexType,ValueType,MemorySpace>& C)
{
    cusp::detail::generic::multiply(A,B,C);
}
    
template <typename ValueType,
          typename MemorySpace>
void multiply(const cusp::array2d<ValueType,MemorySpace>& A,
              const cusp::array2d<ValueType,MemorySpace>& B,
                    cusp::array2d<ValueType,MemorySpace>& C)
{
    cusp::detail::generic::multiply(A,B,C);
}

//////////////////////////////////
// Matrix-Vector Multiplication //
//////////////////////////////////
template <typename IndexType,
          typename ValueType,
          typename MemorySpace1,
          typename MemorySpace2,
          typename MemorySpace3>
void multiply(const cusp::coo_matrix<IndexType,ValueType,MemorySpace1>& A,
              const cusp::array1d<ValueType,MemorySpace2>& B,
                    cusp::array1d<ValueType,MemorySpace3>& C)
{
#ifdef CUSP_USE_TEXTURE_MEMORY    
    cusp::detail::device::spmv_coo_flat_tex(A, thrust::raw_pointer_cast(&B[0]), thrust::raw_pointer_cast(&C[0]));
#else
    cusp::detail::device::spmv_coo_flat(A, thrust::raw_pointer_cast(&B[0]), thrust::raw_pointer_cast(&C[0]));
#endif    
}

template <typename IndexType,
          typename ValueType,
          typename MemorySpace1,
          typename MemorySpace2,
          typename MemorySpace3>
void multiply(const cusp::csr_matrix<IndexType,ValueType,MemorySpace1>& A,
              const cusp::array1d<ValueType,MemorySpace2>& B,
                    cusp::array1d<ValueType,MemorySpace3>& C)
{
#ifdef CUSP_USE_TEXTURE_MEMORY    
    cusp::detail::device::spmv_csr_vector_tex(A, thrust::raw_pointer_cast(&B[0]), thrust::raw_pointer_cast(&C[0]));
#else
    cusp::detail::device::spmv_csr_vector(A, thrust::raw_pointer_cast(&B[0]), thrust::raw_pointer_cast(&C[0]));
#endif    
}

template <typename IndexType,
          typename ValueType,
          typename MemorySpace1,
          typename MemorySpace2,
          typename MemorySpace3>
void multiply(const cusp::dia_matrix<IndexType,ValueType,MemorySpace1>& A,
              const cusp::array1d<ValueType,MemorySpace2>& B,
                    cusp::array1d<ValueType,MemorySpace3>& C)
{
#ifdef CUSP_USE_TEXTURE_MEMORY    
    cusp::detail::device::spmv_dia_tex(A, thrust::raw_pointer_cast(&B[0]), thrust::raw_pointer_cast(&C[0]));
#else
    cusp::detail::device::spmv_dia(A, thrust::raw_pointer_cast(&B[0]), thrust::raw_pointer_cast(&C[0]));
#endif    
}

template <typename IndexType,
          typename ValueType,
          typename MemorySpace1,
          typename MemorySpace2,
          typename MemorySpace3>
void multiply(const cusp::ell_matrix<IndexType,ValueType,MemorySpace1>& A,
              const cusp::array1d<ValueType,MemorySpace2>& B,
                    cusp::array1d<ValueType,MemorySpace3>& C)
{
#ifdef CUSP_USE_TEXTURE_MEMORY    
    cusp::detail::device::spmv_ell_tex(A, thrust::raw_pointer_cast(&B[0]), thrust::raw_pointer_cast(&C[0]));
#else
    cusp::detail::device::spmv_ell(A, thrust::raw_pointer_cast(&B[0]), thrust::raw_pointer_cast(&C[0]));
#endif    
}

template <typename IndexType,
          typename ValueType,
          typename MemorySpace1,
          typename MemorySpace2,
          typename MemorySpace3>
void multiply(const cusp::hyb_matrix<IndexType,ValueType,MemorySpace1>& A,
              const cusp::array1d<ValueType,MemorySpace2>& B,
                    cusp::array1d<ValueType,MemorySpace3>& C)
{
#ifdef CUSP_USE_TEXTURE_MEMORY    
    cusp::detail::device::spmv_tex(A, thrust::raw_pointer_cast(&B[0]), thrust::raw_pointer_cast(&C[0]));
#else
    cusp::detail::device::spmv(A, thrust::raw_pointer_cast(&B[0]), thrust::raw_pointer_cast(&C[0]));
#endif    
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

