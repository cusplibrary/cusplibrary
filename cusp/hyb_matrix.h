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
    template <typename IndexType, typename ValueType, class MemorySpace> class ell_matrix;
    template <typename IndexType, typename ValueType, class MemorySpace> class coo_matrix;

/*! \p hyb_matrix : Hybrid ELL/COO matrix format
 *
 * The \p hyb_matrix is a combination of the \p ell_matrix and
 * \p coo_matrix formats.  Specifically, the \p hyb_matrix format
 * splits a matrix into two portions, one stored in ELL format 
 * and one stored in COO format.
 *
 * While the ELL format is well-suited to vector and SIMD
 * architectures, its efficiency rapidly degrades when the number of
 * nonzeros per matrix row varies.  In contrast, the storage efficiency of
 * the COO format is invariant to the distribution of nonzeros per row, and
 * the use of segmented reduction makes its performance largely invariant
 * as well.  To obtain the advantages of both, we combine these
 * into a hybrid ELL/COO format.
 *
 * The purpose of the HYB format is to store the typical number of
 * nonzeros per row in the ELL data structure and the remaining entries of
 * exceptional rows in the COO format.  
 *
 * \tparam IndexType Type used for matrix indices (e.g. \c int).
 * \tparam ValueType Type used for matrix values (e.g. \c float).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or cusp::device_memory)
 *
 * \note The \p ell_matrix entries must be sorted by column index.
 * \note The \p ell_matrix entries within each row should be shifted to the left.
 * \note The \p coo_matrix entries must be sorted by row index.
 * \note The matrix should not contain duplicate entries.
 *
 *  The following code snippet demonstrates how to create a \p hyb_matrix.
 *  In practice we usually do not construct the HYB format directly and
 *  instead convert from a simpler format such as (COO, CSR) into HYB.
 *
 *  \code
 *  #include <cusp/hyb_matrix.h>
 *  ...
 *
 *  // allocate storage for (4,3) matrix with 8 nonzeros
 *  //     ELL portion has 5 nonzeros and storage for 2 nonzeros per row
 *  //     COO portion has 3 nonzeros
 *
 *  cusp::hyb_matrix<int, float, cusp::host_memory> A(3, 4, 5, 3, 2);
 *  
 *  // Initialize A to represent the following matrix
 *  // [10  20  30  40]
 *  // [ 0  50   0   0]
 *  // [60   0  70  80]
 *  
 *  // A is split into ELL and COO parts as follows
 *  // [10  20  30  40]    [10  20   0   0]     [ 0   0  30  40] 
 *  // [ 0  50   0   0]  = [ 0  50   0   0]  +  [ 0   0   0   0]
 *  // [60   0  70  80]    [60   0  70   0]     [ 0   0   0  80]
 *  
 *  
 *  // Initialize ELL part
 *
 *  // X is used to fill unused entries in the ELL portion of the matrix 
 *  const int X = cusp::ell_matrix<int,float,cusp::host_memory>::invalid_index;
 *
 *  // first row
 *  A.ell.column_indices(0,0) = 0; A.ell.values(0,0) = 10;
 *  A.ell.column_indices(0,1) = 1; A.ell.values(0,1) = 20;
 *
 *  // second row
 *  A.ell.column_indices(1,0) = 1; A.ell.values(1,0) = 50;  // shifted to leftmost position 
 *  A.ell.column_indices(1,1) = X; A.ell.values(1,1) =  0;  // padding
 *
 *  // third row
 *  A.ell.column_indices(2,0) = 0; A.ell.values(2,0) = 60;
 *  A.ell.column_indices(2,1) = 2; A.ell.values(2,1) = 70;  // shifted to leftmost position 
 *
 *
 *  // Initialize COO part
 *  A.coo.row_indices[0] = 0;  A.coo.column_indices[0] = 2;  A.coo.values[0] = 30;
 *  A.coo.row_indices[1] = 0;  A.coo.column_indices[1] = 3;  A.coo.values[1] = 40;
 *  A.coo.row_indices[2] = 2;  A.coo.column_indices[2] = 3;  A.coo.values[2] = 80;
 *
 *  \endcode
 *
 *  \see \p ell_matrix
 *  \see \p coo_matrix
 */
    template <typename IndexType, typename ValueType, class MemorySpace>
    class hyb_matrix : public detail::matrix_base<IndexType,ValueType,MemorySpace>
    {
        public:
        template<typename MemorySpace2>
        struct rebind { typedef hyb_matrix<IndexType, ValueType, MemorySpace2> type; };

        /*! Storage for the \p ell_matrix portion.
         */
        cusp::ell_matrix<IndexType,ValueType,MemorySpace> ell;
        
        /*! Storage for the \p coo_matrix portion.
         */
        cusp::coo_matrix<IndexType,ValueType,MemorySpace> coo;

        /*! Construct an empty \p hyb_matrix.
         */
        hyb_matrix();

        /*! Construct a \p hyb_matrix with a specific shape and separation into ELL and COO portions.
         *
         *  \param num_rows Number of rows.
         *  \param num_cols Number of columns.
         *  \param num_ell_entries Number of nonzero matrix entries in the ELL portion.
         *  \param num_coo_entries Number of nonzero matrix entries in the ELL portion.
         *  \param num_entries_per_row Maximum number of nonzeros per row in the ELL portion.
         *  \param alignment Amount of padding used to align the ELL data structure (default 32).
         */
        hyb_matrix(IndexType num_rows, IndexType num_cols,
                   IndexType num_ell_entries, IndexType num_coo_entries,
                   IndexType num_entries_per_row, IndexType alignment = 32);

        /*! Construct a \p hyb_matrix from another \p hyb_matrix.
         *
         *  \param matrix Another \p hyb_matrix.
         */
        template <typename IndexType2, typename ValueType2, typename MemorySpace2>
        hyb_matrix(const hyb_matrix<IndexType2, ValueType2, MemorySpace2>& matrix);
        
        /*! Construct a \p hyb_matrix from another matrix format.
         *
         *  \param matrix Another sparse or dense matrix.
         */
        template <typename MatrixType>
        hyb_matrix(const MatrixType& matrix);
        
        void resize(IndexType num_rows, IndexType num_cols,
                    IndexType num_ell_entries, IndexType num_coo_entries,
                    IndexType num_entries_per_row, IndexType alignment = 32);

        /*! Swap the contents of two \p hyb_matrix objects.
         *
         *  \param matrix Another \p hyb_matrix with the same IndexType and ValueType.
         */
        void swap(hyb_matrix& matrix);
        
        /*! Assignment from another \p hyb_matrix.
         *
         *  \param matrix Another \p hyb_matrix with possibly different IndexType and ValueType.
         */
        template <typename IndexType2, typename ValueType2, typename MemorySpace2>
        hyb_matrix& operator=(const hyb_matrix<IndexType2, ValueType2, MemorySpace2>& matrix);

        /*! Assignment from another matrix format.
         *
         *  \param matrix Another sparse or dense matrix.
         */
        template <typename MatrixType>
        hyb_matrix& operator=(const MatrixType& matrix);
    }; // class hyb_matrix
/*! \}
 */

} // end namespace cusp

#include <cusp/detail/hyb_matrix.inl>

