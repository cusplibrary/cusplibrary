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


#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/binary_search.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>

namespace cusp
{
namespace precond
{
namespace detail
{
    template <typename IndexType, typename ValueType, typename SpaceOrAlloc,
              typename ArrayType>
        void expand_row_offsets(const cusp::csr_matrix<IndexType,ValueType,SpaceOrAlloc>& A, ArrayType& output)
        {
            output.resize(A.num_entries);

            // compute the row index for each matrix entry
            thrust::upper_bound(A.row_offsets.begin() + 1, A.row_offsets.end(),
                                thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(A.num_entries),
                                output.begin());
        }

    template <typename IndexType, typename ValueType, typename SpaceOrAlloc,
              typename ArrayType>
        void extract_diagonal(const cusp::csr_matrix<IndexType,ValueType,SpaceOrAlloc>& A, ArrayType& output)
        {
            output.resize(thrust::min(A.num_rows,A.num_cols));

            // initialize output to zero
            thrust::fill(output.begin(), output.end(), ValueType(0));

            // first expand the compressed row offsets into row indices
            cusp::array1d<IndexType,SpaceOrAlloc> row_indices;
            expand_row_offsets(A, row_indices);

            // determine which matrix entries correspond to the matrix diagonal
            cusp::array1d<unsigned int,SpaceOrAlloc> is_diagonal(A.num_entries);
            thrust::transform(row_indices.begin(), row_indices.end(), A.column_indices.begin(), is_diagonal.begin(), thrust::equal_to<IndexType>());

            // scatter the diagonal values to output
            thrust::scatter_if(A.values.begin(), A.values.end(),
                               row_indices.begin(),
                               is_diagonal.begin(),
                               output.begin());
        }

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
    template<typename IndexType, typename ValueType2, class SpaceOrAlloc>
    diagonal<ValueType, MemorySpace>
    ::diagonal(const cusp::csr_matrix<IndexType, ValueType2, SpaceOrAlloc>& A)
    {
        // extract the main diagonal
        detail::extract_diagonal(A, diagonal_reciprocals);

        // invert the entries
        thrust::transform(diagonal_reciprocals.begin(), diagonal_reciprocals.end(),
                          diagonal_reciprocals.begin(), detail::reciprocal<ValueType>());
    }
        
template <typename ValueType, typename MemorySpace>
    template <typename VectorType1, typename VectorType2>
    void diagonal<ValueType, MemorySpace>
    ::multiply(const VectorType1& x, VectorType2& y) const
    {
        cusp::blas::xmy(diagonal_reciprocals, x, y);
    }

} // end namespace precond
} // end namespace cusp

