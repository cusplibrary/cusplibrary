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

#include <cusp/format.h>

#include <cusp/system/detail/sequential/execution_policy.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace sequential
{

////////////////////////////////////////
// Dense Matrix-Matrix Multiplication //
////////////////////////////////////////
template <typename Matrix1,
         typename Matrix2,
         typename Matrix3>
void multiply(const Matrix1&  A,
              const Matrix2& B,
              Matrix3& C,
              cusp::array2d_format,
              cusp::array2d_format,
              cusp::array2d_format)
{
    typedef typename Matrix3::value_type ValueType;

    C.resize(A.num_rows, B.num_cols);

    for(size_t i = 0; i < C.num_rows; i++)
    {
        for(size_t j = 0; j < C.num_cols; j++)
        {
            ValueType v = 0;

            for(size_t k = 0; k < A.num_cols; k++)
                v += A(i,k) * B(k,j);

            C(i,j) = v;
        }
    }
}

/////////////////////////////////////////
// Sparse Matrix-Matrix Multiplication //
/////////////////////////////////////////
template <typename Matrix1,
         typename Matrix2,
         typename Matrix3>
void multiply(const Matrix1& A,
              const Matrix2& B,
              Matrix3& C,
              cusp::coo_format,
              cusp::coo_format,
              cusp::coo_format)
{
    cusp::detail::host::detail::spmm_coo(A,B,C);
}

template <typename Matrix1,
         typename Matrix2,
         typename Matrix3>
void multiply(const Matrix1& A,
              const Matrix2& B,
              Matrix3& C,
              cusp::csr_format,
              cusp::csr_format,
              cusp::csr_format)
{
    cusp::detail::host::detail::spmm_csr(A,B,C);
}

} // end namespace sequential
} // end namespace detail
} // end namespace system
} // end namespace cusp

