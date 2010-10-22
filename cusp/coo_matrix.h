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

/*! \file coo_matrix.h
 *  \brief Coordinate matrix format.
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/format.h>
#include <cusp/array1d.h>
#include <cusp/detail/matrix_base.h>

namespace cusp
{

/*! \addtogroup containers Containers 
 *  \addtogroup sparse_matrix_formats Sparse Matrices
 *  \ingroup containers
 *  \{
 */

/*! \p coo_matrix : Coordinate matrix container
 *
 * \tparam IndexType Type used for matrix indices (e.g. \c int).
 * \tparam ValueType Type used for matrix values (e.g. \c float).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or cusp::device_memory)
 *
 * \note The matrix entries must be sorted by row index.
 * \note The matrix should not contain duplicate entries.
 *
 *  The following code snippet demonstrates how to create a 4-by-3
 *  \p coo_matrix on the host with 6 nonzeros and then copies the
 *  matrix to the device.
 *
 *  \code
 *  #include <cusp/coo_matrix.h>
 *  ...
 *
 *  // allocate storage for (4,3) matrix with 6 nonzeros
 *  cusp::coo_matrix<int,float,cusp::host_memory> A(4,3,6);
 *
 *  // initialize matrix entries on host
 *  A.row_indices[0] = 0; A.column_indices[0] = 0; A.values[0] = 10;
 *  A.row_indices[1] = 0; A.column_indices[1] = 2; A.values[1] = 20;
 *  A.row_indices[2] = 2; A.column_indices[2] = 2; A.values[2] = 30;
 *  A.row_indices[3] = 3; A.column_indices[3] = 0; A.values[3] = 40;
 *  A.row_indices[4] = 3; A.column_indices[4] = 1; A.values[4] = 50;
 *  A.row_indices[5] = 3; A.column_indices[5] = 2; A.values[5] = 60;
 *
 *  // A now represents the following matrix
 *  //    [10  0 20]
 *  //    [ 0  0  0]
 *  //    [ 0  0 30]
 *  //    [40 50 60]
 *
 *  // copy to the device
 *  cusp::coo_matrix<int,float,cusp::device_memory> B = A;
 *  \endcode
 *
 */
template <typename IndexType, typename ValueType, class MemorySpace>
class coo_matrix : public detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::coo_format>
{
    public:
    template<typename MemorySpace2>
    struct rebind { typedef coo_matrix<IndexType, ValueType, MemorySpace2> type; };

    /*! Storage for the row indices of the COO data structure.
     */
    cusp::array1d<IndexType, MemorySpace> row_indices;
    
    /*! Storage for the column indices of the COO data structure.
     */
    cusp::array1d<IndexType, MemorySpace> column_indices;

    /*! Storage for the nonzero entries of the COO data structure.
     */
    cusp::array1d<ValueType, MemorySpace> values;

    /*! Construct an empty \p coo_matrix.
     */
    coo_matrix();

    /*! Construct a \p coo_matrix with a specific shape and number of nonzero entries.
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero matrix entries.
     */
    coo_matrix(IndexType num_rows, IndexType num_cols, IndexType num_entries);

    /*! Construct a \p coo_matrix from another matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template <typename MatrixType>
    coo_matrix(const MatrixType& matrix);

    void resize(IndexType num_rows, IndexType num_cols, IndexType num_entries);

    /*! Swap the contents of two \p coo_matrix objects.
     *
     *  \param matrix Another \p coo_matrix with the same IndexType and ValueType.
     */
    void swap(coo_matrix& matrix);

    /*! Assignment from another matrix.
     *
     *  \param matrix Another sparse or dense matrix.
     */
    template <typename MatrixType>
    coo_matrix& operator=(const MatrixType& matrix);

    /*! Sort matrix elements by row index
     */
    void sort_by_row(void);
    
    /*! Determine whether matrix elements are sorted by row index
     *
     *  \return \c false, if the row indices are unsorted; \c true, otherwise.
     */
    bool is_sorted_by_row(void);
}; // class coo_matrix
/*! \}
 */


/*! \p coo_matrix_view : Coordinate matrix view
 *
 * TODO document
 *
 */
template <typename Array1,
          typename Array2,
          typename Array3,
          typename IndexType   = typename Array1::value_type,
          typename ValueType   = typename Array3::value_type,
          typename MemorySpace = typename Array1::memory_space> // TODO use minimum_space<1,2,3>::type
          class coo_matrix_view : public cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::coo_format>
{
  typedef cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::coo_format> Parent;
  public:
    /*! View of the row indices of the COO data structure.  Also called the "row pointer" array.
     */
    Array1 row_indices;
    
    /*! View of the column indices of the COO data structure.
     */
    Array2 column_indices;
    
    /*! View for the nonzero entries of the COO data structure.
     */
    Array3 values;

    // construct empty view
    coo_matrix_view(void)
      : Parent() {}

    // construct from existing COO matrix or view
    template <typename Matrix>
    coo_matrix_view(Matrix& A)
      : Parent(A),
        row_indices(A.row_indices),
        column_indices(A.column_indices),
        values(A.values) {}
  
    // TODO check sizes here
    coo_matrix_view(IndexType num_rows,
                    IndexType num_cols,
                    IndexType num_entries,
                    Array1 row_indices,
                    Array2 column_indices,
                    Array3 values)
      : Parent(num_rows, num_cols, num_entries),
        row_indices(row_indices),
        column_indices(column_indices),
        values(values) {}

    void resize(IndexType num_rows, IndexType num_cols, IndexType num_entries)
    {
      Parent::resize(num_rows, num_cols, num_entries);
      row_indices.resize(num_entries);
      column_indices.resize(num_entries);
      values.resize(num_entries);
    }
};

/* Convenience functions */

template <typename IndexType,
          typename Array1,
          typename Array2,
          typename Array3>
coo_matrix_view<Array1,Array2,Array3,IndexType>
make_coo_matrix_view(IndexType num_rows,
                     IndexType num_cols,
                     IndexType num_entries,
                     Array1 row_indices,
                     Array2 column_indices,
                     Array3 values)
{
  return coo_matrix_view<Array1,Array2,Array3,IndexType>
    (num_rows, num_cols, num_entries,
     row_indices, column_indices, values);
}

template <typename Array1,
          typename Array2,
          typename Array3,
          typename IndexType,
          typename ValueType,
          typename MemorySpace>
coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>
make_coo_matrix_view(const coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>& m)
{
  return coo_matrix_view<Array1,Array2,Array3,IndexType,ValueType,MemorySpace>(m);
}
    
template <typename IndexType, typename ValueType, class MemorySpace>
coo_matrix_view < cusp::array1d_view< typename cusp::array1d<IndexType,MemorySpace>::iterator >,
                  cusp::array1d_view< typename cusp::array1d<IndexType,MemorySpace>::iterator >,
                  cusp::array1d_view< typename cusp::array1d<ValueType,MemorySpace>::iterator >,
                  IndexType, ValueType, MemorySpace>
make_coo_matrix_view(coo_matrix<IndexType,ValueType,MemorySpace>& m)
{
  return make_coo_matrix_view
    (m.num_rows, m.num_cols, m.num_entries,
     make_array1d_view(m.row_indices),
     make_array1d_view(m.column_indices),
     make_array1d_view(m.values));
}

template <typename IndexType, typename ValueType, class MemorySpace>
coo_matrix_view < cusp::array1d_view< typename cusp::array1d<IndexType,MemorySpace>::const_iterator >,
                  cusp::array1d_view< typename cusp::array1d<IndexType,MemorySpace>::const_iterator >,
                  cusp::array1d_view< typename cusp::array1d<ValueType,MemorySpace>::const_iterator >,
                  IndexType, ValueType, MemorySpace>
make_coo_matrix_view(const coo_matrix<IndexType,ValueType,MemorySpace>& m)
{
  return make_coo_matrix_view
    (m.num_rows, m.num_cols, m.num_entries,
     make_array1d_view(m.row_indices),
     make_array1d_view(m.column_indices),
     make_array1d_view(m.values));
}
/*! \}
 */


} // end namespace cusp

#include <cusp/detail/coo_matrix.inl>

