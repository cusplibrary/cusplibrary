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

#pragma once

#include <cusp/detail/config.h>

#include <cusp/detail/format.h>
#include <cusp/format_utils.h>

#include <cusp/system/omp/detail/multiply/csr_spgemm.h>

namespace cusp
{
namespace system
{
namespace omp
{

template <typename DerivedPolicy,
         typename Matrix1,
         typename Matrix2,
         typename Matrix3>
void multiply(omp::execution_policy<DerivedPolicy>& exec,
              const Matrix1& A,
              const Matrix2& B,
              Matrix3& C,
              coo_format,
              coo_format,
              coo_format)
{
    // allocate storage for row offsets for A, B, and C
    cusp::array1d<typename Matrix1::index_type,cusp::host_memory> A_row_offsets(A.num_rows + 1);
    cusp::array1d<typename Matrix2::index_type,cusp::host_memory> B_row_offsets(B.num_rows + 1);
    cusp::array1d<typename Matrix3::index_type,cusp::host_memory> C_row_offsets(A.num_rows + 1);

    // compute row offsets for A and B
    cusp::indices_to_offsets(A.row_indices, A_row_offsets);
    cusp::indices_to_offsets(B.row_indices, B_row_offsets);

    typedef typename Matrix3::index_type IndexType;

    IndexType estimated_nonzeros =
        spmm_csr_pass1(A.num_rows, B.num_cols,
                       A_row_offsets, A.column_indices,
                       B_row_offsets, B.column_indices);

    // Resize output
    C.resize(A.num_rows, B.num_cols, estimated_nonzeros);

    IndexType true_nonzeros =
        spmm_csr_pass2(A.num_rows, B.num_cols,
                       A_row_offsets, A.column_indices, A.values,
                       B_row_offsets, B.column_indices, B.values,
                       C_row_offsets, C.column_indices, C.values);

    // true_nonzeros may be less than estimated_nonzeros
    C.resize(A.num_rows, B.num_cols, true_nonzeros);

    cusp::offsets_to_indices(C_row_offsets, C.row_indices);
}

} // end namespace omp
} // end namespace system
} // end namespace cusp

