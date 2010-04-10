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

/*! \file dia_matrix.h
 *  \brief Diagonal matrix format.
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

    // Forward definitions
    struct column_major;
    template<typename ValueType, class MemorySpace, class Orientation> class array2d;

/*! \p dia_matrix : Diagonal matrix format
 *
 * \tparam IndexType Type used for matrix indices (e.g. \c int).
 * \tparam ValueType Type used for matrix values (e.g. \c float).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or cusp::device_memory)
 *
 * \note The diagonal offsets should not contain duplicate entries.
 *
 *  The following code snippet demonstrates how to create a 4-by-3
 *  \p dia_matrix on the host with 3 diagonals (6 total nonzeros)
 *  and then copies the matrix to the device.
 *
 *  \code
 *  #include <cusp/dia_matrix.h>
 *  ...
 *
 *  // allocate storage for (4,3) matrix with 6 nonzeros in 3 diagonals
 *  cusp::dia_matrix<int,float,cusp::host_memory> A(4,3,6,3);
 *
 *  // initialize diagonal offsets
 *  A.diagonal_offsets[0] = -2;
 *  A.diagonal_offsets[1] =  0;
 *  A.diagonal_offsets[2] =  1;
 *
 *  // initialize diagonal values
 *
 *  // first diagonal
 *  A.values(0,2) =  0;  // outside matrix
 *  A.values(1,2) =  0;  // outside matrix
 *  A.values(2,0) = 40;
 *  A.values(3,0) = 60;
 *  
 *  // second diagonal
 *  A.values(0,1) = 10;
 *  A.values(1,1) =  0;
 *  A.values(2,1) = 50;
 *  A.values(3,1) = 50;  // outside matrix
 *
 *  // third diagonal
 *  A.values(0,2) = 20;
 *  A.values(1,2) = 30;
 *  A.values(2,2) =  0;  // outside matrix
 *  A.values(3,2) =  0;  // outside matrix
 *
 *  // A now represents the following matrix
 *  //    [10 20  0]
 *  //    [ 0  0 30]
 *  //    [40  0 50]
 *  //    [ 0 60  0]
 *
 *  // copy to the device
 *  cusp::dia_matrix<int,float,cusp::device_memory> B = A;
 *  \endcode
 *
 */
    template <typename IndexType, typename ValueType, class MemorySpace>
    class dia_matrix : public detail::matrix_base<IndexType,ValueType,MemorySpace>
    {
        public:
        // TODO statically assert is_signed<IndexType>
        
        template<typename MemorySpace2>
        struct rebind { typedef dia_matrix<IndexType, ValueType, MemorySpace2> type; };

        /*! Storage for the diagonal offsets.
         */
        cusp::array1d<IndexType, MemorySpace> diagonal_offsets;
        
        /*! Storage for the nonzero entries of the DIA data structure.
         */
        cusp::array2d<ValueType, MemorySpace, cusp::column_major> values;
            
        /*! Construct an empty \p dia_matrix.
         */
        dia_matrix();

        /*! Construct a \p dia_matrix with a specific shape, number of nonzero entries,
         *  and number of occupied diagonals.
         *
         *  \param num_rows Number of rows.
         *  \param num_cols Number of columns.
         *  \param num_entries Number of nonzero matrix entries.
         *  \param num_diagonals Number of occupied diagonals.
         *  \param alignment Amount of padding used to align the data structure (default 32).
         */
        dia_matrix(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                   IndexType num_diagonals, IndexType alignment = 32);
        
        /*! Construct a \p dia_matrix from another \p dia_matrix.
         *
         *  \param matrix Another \p dia_matrix.
         */
        template <typename IndexType2, typename ValueType2, typename MemorySpace2>
        dia_matrix(const dia_matrix<IndexType2, ValueType2, MemorySpace2>& matrix);
        
        /*! Construct a \p dia_matrix from another matrix format.
         *
         *  \param matrix Another sparse or dense matrix.
         */
        template <typename MatrixType>
        dia_matrix(const MatrixType& matrix);
        
        void resize(IndexType num_rows, IndexType num_cols, IndexType num_entries,
                    IndexType num_diagonals, IndexType alignment = 32);

        /*! Swap the contents of two \p dia_matrix objects.
         *
         *  \param matrix Another \p dia_matrix with the same IndexType and ValueType.
         */
        void swap(dia_matrix& matrix);
        
        /*! Assignment from another \p dia_matrix.
         *
         *  \param matrix Another \p dia_matrix with possibly different IndexType and ValueType.
         */
        template <typename IndexType2, typename ValueType2, typename MemorySpace2>
        dia_matrix& operator=(const dia_matrix<IndexType2, ValueType2, MemorySpace2>& matrix);

        /*! Assignment from another matrix format.
         *
         *  \param matrix Another sparse or dense matrix.
         */
        template <typename MatrixType>
        dia_matrix& operator=(const MatrixType& matrix);
    }; // class dia_matrix
/*! \}
 */
    
} // end namespace cusp

#include <cusp/array2d.h>

#include <cusp/detail/dia_matrix.inl>

