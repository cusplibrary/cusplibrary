/*
 *  Copyright 2008-2014 NVIDIA Corporation
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

/*! \file sort.h
 *  \brief Specialized sorting routines
 */

#pragma once

#include <cusp/detail/config.h>

#include <thrust/execution_policy.h>

namespace cusp
{

/*! \addtogroup algorithms Algorithms
 *  \addtogroup matrix_algorithms Matrix Algorithms
 *  \ingroup algorithms
 *  \{
 */

/* \cond */
template <typename DerivedPolicy, typename ArrayType1, typename ArrayType2, typename ArrayType3>
void sort_by_row(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                 ArrayType1& row_indices, ArrayType2& column_indices, ArrayType3& values);
/* \endcond */

/**
 * \brief Sort matrix indices by row
 *
 * \tparam ArrayType1 Type of input matrix row indices
 * \tparam ArrayType2 Type of input matrix column indices
 * \tparam ArrayType3 Type of input matrix values
 *
 * \param row_indices input matrix row indices
 * \param column_indices input matrix column indices
 * \param values input matrix values
 *
 * \par Example
 *  The following code snippet demonstrates how to use \p sort_by_row.
 *
 *  \code
 *  #include <cusp/coo_matrix.h>
 *  #include <cusp/print.h>
 *  #include <cusp/sort.h>
 *
 *  int main(void)
 *  {
 *      // allocate storage for (4,3) matrix with 6 nonzeros
 *      cusp::coo_matrix<int,float,cusp::host_memory> A(4,3,6);
 *
 *      // initialize matrix entries on host
 *      A.row_indices[0] = 3; A.column_indices[0] = 0; A.values[0] = 10;
 *      A.row_indices[1] = 3; A.column_indices[1] = 2; A.values[1] = 20;
 *      A.row_indices[2] = 2; A.column_indices[2] = 0; A.values[2] = 30;
 *      A.row_indices[3] = 2; A.column_indices[3] = 2; A.values[3] = 40;
 *      A.row_indices[4] = 0; A.column_indices[4] = 1; A.values[4] = 50;
 *      A.row_indices[5] = 0; A.column_indices[5] = 2; A.values[5] = 60;
 *
 *      // sort A by row
 *      cusp::sort_by_row(A.row_indices, A.column_indices, A.values);
 *
 *      return 0;
 *  }
 *  \endcode
 */
template <typename ArrayType1, typename ArrayType2, typename ArrayType3>
void sort_by_row(ArrayType1& rows, ArrayType2& columns, ArrayType3& values);

/* \cond */
template <typename DerivedPolicy, typename ArrayType1, typename ArrayType2, typename ArrayType3>
void sort_by_row_and_column(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                            ArrayType1& row_indices, ArrayType2& column_indices, ArrayType3& values);
/* \endcond */

/**
 * \brief Sort matrix indices by row and column
 *
 * \tparam ArrayType1 Type of input matrix row indices
 * \tparam ArrayType2 Type of input matrix column indices
 * \tparam ArrayType3 Type of input matrix values
 *
 * \param row_indices input matrix row indices
 * \param column_indices input matrix column indices
 * \param values input matrix values
 *
 * \par Example
 *  The following code snippet demonstrates how to use \p
 *  sort_by_row_and_column.
 *
 *  \code
 *  #include <cusp/coo_matrix.h>
 *  #include <cusp/print.h>
 *  #include <cusp/sort.h>
 *
 *  int main(void)
 *  {
 *      // allocate storage for (4,3) matrix with 6 nonzeros
 *      cusp::coo_matrix<int,float,cusp::host_memory> A(4,3,6);
 *
 *      // initialize matrix entries on host
 *      A.row_indices[0] = 3; A.column_indices[0] = 2; A.values[0] = 10;
 *      A.row_indices[1] = 3; A.column_indices[1] = 0; A.values[1] = 20;
 *      A.row_indices[2] = 2; A.column_indices[2] = 0; A.values[2] = 30;
 *      A.row_indices[3] = 2; A.column_indices[3] = 2; A.values[3] = 40;
 *      A.row_indices[4] = 0; A.column_indices[4] = 2; A.values[4] = 50;
 *      A.row_indices[5] = 0; A.column_indices[5] = 1; A.values[5] = 60;
 *
 *      // sort A by row
 *      cusp::sort_by_row_and_column(A.row_indices, A.column_indices, A.values);
 *
 *      return 0;
 *  }
 *  \endcode
 */
template <typename ArrayType1, typename ArrayType2, typename ArrayType3>
void sort_by_row_and_column(ArrayType1& rows, ArrayType2& columns, ArrayType3& values);
/*! \}
 */

} // end namespace cusp

#include <cusp/detail/sort.inl>

