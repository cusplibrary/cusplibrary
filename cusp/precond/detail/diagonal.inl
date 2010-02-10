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

/*! \file diagonal.inl
 *  \brief Inline file for diagonal.h
 */

#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/blas.h>
#include <cusp/detail/format_utils.h>

#include <thrust/functional.h>
#include <thrust/transform.h>

namespace cusp
{
namespace precond
{
namespace detail
{
    template <typename T>
        struct reciprocal : public thrust::unary_function<T,T>
    {
        T operator()(const T& v)
        {
            return T(1.0) / v;
        }
    };

} // end namespace detail
    
template <typename ValueType, typename MemorySpace>
    template<typename IndexType2, typename ValueType2, class MemorySpace2>
    diagonal<ValueType, MemorySpace>
    ::diagonal(const cusp::csr_matrix<IndexType2, ValueType2, MemorySpace2>& A)
    {
        // extract the main diagonal
        cusp::detail::extract_diagonal(A, diagonal_reciprocals);

        // invert the entries
        thrust::transform(diagonal_reciprocals.begin(), diagonal_reciprocals.end(),
                          diagonal_reciprocals.begin(), detail::reciprocal<ValueType>());
    }
        
template <typename ValueType, typename MemorySpace>
    template<typename IndexType2, typename ValueType2, class MemorySpace2>
    diagonal<ValueType, MemorySpace>
    ::diagonal(const cusp::coo_matrix<IndexType2, ValueType2, MemorySpace2>& A)
    : diagonal_reciprocals(thrust::min(A.num_rows,A.num_cols), ValueType(0))
    {
        // extract the main diagonal
        cusp::detail::extract_diagonal(A, diagonal_reciprocals);

        // invert the entries
        thrust::transform(diagonal_reciprocals.begin(), diagonal_reciprocals.end(),
                          diagonal_reciprocals.begin(), detail::reciprocal<ValueType>());
    }
        
template <typename ValueType, typename MemorySpace>
    template<typename IndexType2, typename ValueType2, class MemorySpace2>
    diagonal<ValueType, MemorySpace>
    ::diagonal(const cusp::ell_matrix<IndexType2, ValueType2, MemorySpace2>& A)
    : diagonal_reciprocals(thrust::min(A.num_rows,A.num_cols), ValueType(0))
    {
        // extract the main diagonal
        cusp::detail::extract_diagonal(A, diagonal_reciprocals);

        // invert the entries
        thrust::transform(diagonal_reciprocals.begin(), diagonal_reciprocals.end(),
                          diagonal_reciprocals.begin(), detail::reciprocal<ValueType>());
    }
        
template <typename ValueType, typename MemorySpace>
    template<typename IndexType2, typename ValueType2, class MemorySpace2>
    diagonal<ValueType, MemorySpace>
    ::diagonal(const cusp::hyb_matrix<IndexType2, ValueType2, MemorySpace2>& A)
    : diagonal_reciprocals(thrust::min(A.num_rows,A.num_cols), ValueType(0))
    {
        // extract the main diagonal
        cusp::detail::extract_diagonal(A.coo, diagonal_reciprocals);
        cusp::detail::extract_diagonal(A.ell, diagonal_reciprocals);

        // invert the entries
        thrust::transform(diagonal_reciprocals.begin(), diagonal_reciprocals.end(),
                          diagonal_reciprocals.begin(), detail::reciprocal<ValueType>());
    }
        
template <typename ValueType, typename MemorySpace>
    template <typename VectorType1, typename VectorType2>
    void diagonal<ValueType, MemorySpace>
    ::operator()(const VectorType1& x, VectorType2& y) const
    {
        cusp::blas::xmy(diagonal_reciprocals, x, y);
    }

} // end namespace precond
} // end namespace cusp

