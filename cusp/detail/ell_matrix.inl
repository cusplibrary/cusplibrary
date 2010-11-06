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
#include <cusp/detail/utils.h>

namespace cusp
{

//////////////////
// Constructors //
//////////////////
        
// construct empty matrix
template <typename IndexType, typename ValueType, class MemorySpace>
ell_matrix<IndexType,ValueType,MemorySpace>
    ::ell_matrix()
        : detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::ell_format>() {}

// construct matrix with given shape and number of entries
template <typename IndexType, typename ValueType, class MemorySpace>
ell_matrix<IndexType,ValueType,MemorySpace>
    ::ell_matrix(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                 IndexType num_entries_per_row, IndexType alignment)
        : detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::ell_format>(num_rows, num_cols, num_entries)
    {
      // TODO use array2d constructor when it can accept pitch
      column_indices.resize(num_rows, num_entries_per_row, detail::round_up(num_rows, alignment));
      values.resize        (num_rows, num_entries_per_row, detail::round_up(num_rows, alignment));
    }

// construct from a different matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
ell_matrix<IndexType,ValueType,MemorySpace>
    ::ell_matrix(const MatrixType& matrix)
    {
        cusp::convert(matrix, *this);
    }

//////////////////////
// Member Functions //
//////////////////////

// assignment from another matrix
template <typename IndexType, typename ValueType, class MemorySpace>
template <typename MatrixType>
    ell_matrix<IndexType,ValueType,MemorySpace>&
    ell_matrix<IndexType,ValueType,MemorySpace>
    ::operator=(const MatrixType& matrix)
    {
        cusp::convert(matrix, *this);
        
        return *this;
    }

///////////////////////////
// Convenience Functions //
///////////////////////////

template <typename IndexType,
          typename Array1,
          typename Array2>
ell_matrix_view<Array1,Array2,IndexType>
make_ell_matrix_view(IndexType num_rows,
                     IndexType num_cols,
                     IndexType num_entries,
                     Array1 column_indices,
                     Array2 values)
{
  return ell_matrix_view<Array1,Array2,IndexType>
    (num_rows, num_cols, num_entries,
     column_indices, values);
}

template <typename Array1,
          typename Array2,
          typename IndexType,
          typename ValueType,
          typename MemorySpace>
ell_matrix_view<Array1,Array2,IndexType,ValueType,MemorySpace>
make_ell_matrix_view(const ell_matrix_view<Array1,Array2,IndexType,ValueType,MemorySpace>& m)
{
  return ell_matrix_view<Array1,Array2,IndexType,ValueType,MemorySpace>(m);
}
    
template <typename IndexType, typename ValueType, class MemorySpace>
ell_matrix_view <typename cusp::array2d_view<typename cusp::array1d_view<typename cusp::array1d<IndexType,MemorySpace>::iterator >, cusp::column_major>,
                 typename cusp::array2d_view<typename cusp::array1d_view<typename cusp::array1d<ValueType,MemorySpace>::iterator >, cusp::column_major>,
                 IndexType, ValueType, MemorySpace>
make_ell_matrix_view(ell_matrix<IndexType,ValueType,MemorySpace>& m)
{
  return make_ell_matrix_view
    (m.num_rows, m.num_cols, m.num_entries,
     cusp::make_array2d_view(m.column_indices),
     cusp::make_array2d_view(m.values));
}

template <typename IndexType, typename ValueType, class MemorySpace>
ell_matrix_view <typename cusp::array2d_view<typename cusp::array1d_view<typename cusp::array1d<IndexType,MemorySpace>::const_iterator >, cusp::column_major>,
                 typename cusp::array2d_view<typename cusp::array1d_view<typename cusp::array1d<ValueType,MemorySpace>::const_iterator >, cusp::column_major>,
                 IndexType, ValueType, MemorySpace>
make_ell_matrix_view(const ell_matrix<IndexType,ValueType,MemorySpace>& m)
{
  return make_ell_matrix_view
    (m.num_rows, m.num_cols, m.num_entries,
     cusp::make_array2d_view(m.column_indices),
     cusp::make_array2d_view(m.values));
}

} // end namespace cusp

