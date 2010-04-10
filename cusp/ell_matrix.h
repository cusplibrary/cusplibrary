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

/*! \file ell_matrix.h
 *  \brief ELLPACK/ITPACK matrix format.
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/detail/matrix_base.h>

namespace cusp
{
/*! \addtogroup containers Containers 
 *  \addtogroup sparse_matrix_formats Sparse Matrices
 *  \ingroup containers
 *  \{
 */

    // Forward definitions
    struct column_major;
    template<typename ValueType, class MemorySpace, class Orientation> class array2d;

/*! \p ell_matrix : ELLPACK/ITPACK matrix format
 *
 * \tparam IndexType Type used for matrix indices (e.g. \c int).
 * \tparam ValueType Type used for matrix values (e.g. \c float).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or cusp::device_memory)
 *
 * \note The matrix entries must be sorted by column index.
 * \note The matrix entries within each row should be shifted to the left.
 * \note The matrix should not contain duplicate entries.
 *
 *  The following code snippet demonstrates how to create a 4-by-3
 *  \p ell_matrix on the host with 3 nonzeros per row (6 total nonzeros)
 *  and then copies the matrix to the device.
 *
 *  \code
 *  #include <cusp/ell_matrix.h>
 *  ...
 *
 *  // allocate storage for (4,3) matrix with 6 nonzeros and at most 3 nonzeros per row.
 *  cusp::ell_matrix<int,float,cusp::host_memory> A(4,3,6,3);
 *
 *  // X is used to fill unused entries in the matrix
 *  const int X = cusp::ell_matrix<int,float,cusp::host_memory>::invalid_index;
 *
 *  // initialize matrix entries on host
 *  A.column_indices(0,0) = 0; A.values(0,0) = 10;
 *  A.column_indices(0,1) = 2; A.values(0,1) = 20;  // shifted to leftmost position
 *  A.column_indices(0,2) = X; A.values(0,2) =  0;  // padding
 *
 *  A.column_indices(1,0) = X; A.values(1,0) =  0;  // padding
 *  A.column_indices(1,1) = X; A.values(1,1) =  0;  // padding
 *  A.column_indices(1,2) = X; A.values(1,2) =  0;  // padding
 *
 *  A.column_indices(2,0) = 2; A.values(2,0) = 30;  // shifted to leftmost position
 *  A.column_indices(2,1) = X; A.values(2,1) =  0;  // padding
 *  A.column_indices(2,2) = X; A.values(2,2) =  0;  // padding
 *
 *  A.column_indices(3,0) = 0; A.values(3,0) = 40;
 *  A.column_indices(3,1) = 1; A.values(3,1) = 50;
 *  A.column_indices(3,2) = 2; A.values(3,2) = 60;
 *
 *  // A now represents the following matrix
 *  //    [10  0 20]
 *  //    [ 0  0  0]
 *  //    [ 0  0 30]
 *  //    [40 50 60]
 *
 *  // copy to the device
 *  cusp::ell_matrix<int,float,cusp::device_memory> B = A;
 *  \endcode
 *
 */
    template <typename IndexType, typename ValueType, class MemorySpace>
    class ell_matrix : public detail::matrix_base<IndexType,ValueType,MemorySpace>
    {
        public:
        template<typename MemorySpace2>
        struct rebind { typedef ell_matrix<IndexType, ValueType, MemorySpace2> type; };

        /*! Value used to pad the rows of the column_indices array.
         */
        const static IndexType invalid_index = static_cast<IndexType>(-1);
        
        /*! Storage for the column indices of the ELL data structure.
         */
        cusp::array2d<IndexType, MemorySpace, cusp::column_major> column_indices;

        /*! Storage for the nonzero entries of the ELL data structure.
         */
        cusp::array2d<ValueType, MemorySpace, cusp::column_major> values;
    
        /*! Construct an empty \p ell_matrix.
         */
        ell_matrix();
    
        /*! Construct an \p ell_matrix with a specific shape, number of nonzero entries,
         *  and maximum number of nonzero entries per row.
         *
         *  \param num_rows Number of rows.
         *  \param num_cols Number of columns.
         *  \param num_entries Number of nonzero matrix entries.
         *  \param num_entries_per_row Maximum number of nonzeros per row.
         *  \param alignment Amount of padding used to align the data structure (default 32).
         */
        ell_matrix(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                   IndexType num_entries_per_row, IndexType alignment = 32);
    
        /*! Construct an \p ell_matrix from another \p ell_matrix.
         *
         *  \param matrix Another \p ell_matrix.
         */
        template <typename IndexType2, typename ValueType2, typename MemorySpace2>
        ell_matrix(const ell_matrix<IndexType2, ValueType2, MemorySpace2>& matrix);
        
        /*! Construct an \p ell_matrix from another matrix format.
         *
         *  \param matrix Another sparse or dense matrix.
         */
        template <typename MatrixType>
        ell_matrix(const MatrixType& matrix);
        
        void resize(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                    IndexType num_entries_per_row, IndexType alignment = 32);

        /*! Swap the contents of two \p ell_matrix objects.
         *
         *  \param matrix Another \p ell_matrix with the same IndexType and ValueType.
         */
        void swap(ell_matrix& matrix);
        
        /*! Assignment from another \p ell_matrix.
         *
         *  \param matrix Another \p ell_matrix with possibly different IndexType and ValueType.
         */
        template <typename IndexType2, typename ValueType2, typename MemorySpace2>
        ell_matrix& operator=(const ell_matrix<IndexType2, ValueType2, MemorySpace2>& matrix);

        /*! Assignment from another matrix format.
         *
         *  \param matrix Another sparse or dense matrix.
         */
        template <typename MatrixType>
        ell_matrix& operator=(const MatrixType& matrix);
    }; // class ell_matrix
/*! \}
 */

} // end namespace cusp

#include <cusp/array2d.h>
#include <cusp/detail/ell_matrix.inl>

