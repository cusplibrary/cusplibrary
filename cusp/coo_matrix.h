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

#include <cusp/array1d.h>
#include <cusp/detail/matrix_base.h>

namespace cusp
{

/*! \addtogroup containers Containers 
 *  \addtogroup sparse_matrix_formats Sparse Matrices
 *  \ingroup containers
 *  \{
 */

/*! \p coo_matrix : Coordinate matrix format
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
    class coo_matrix : public detail::matrix_base<IndexType,ValueType,MemorySpace>
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
    
        /*! Construct a \p coo_matrix from another \p coo_matrix.
         *
         *  \param matrix Another \p coo_matrix.
         */
        template <typename IndexType2, typename ValueType2, typename MemorySpace2>
        coo_matrix(const coo_matrix<IndexType2, ValueType2, MemorySpace2>& matrix);
        
        /*! Construct a \p coo_matrix from another matrix format.
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

        /*! Assignment from another \p coo_matrix.
         *
         *  \param matrix Another \p coo_matrix with possibly different IndexType and ValueType.
         */
        template <typename IndexType2, typename ValueType2, typename MemorySpace2>
        coo_matrix& operator=(const coo_matrix<IndexType2, ValueType2, MemorySpace2>& matrix);

        /*! Assignment from another matrix format.
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

} // end namespace cusp

#include <cusp/detail/coo_matrix.inl>

