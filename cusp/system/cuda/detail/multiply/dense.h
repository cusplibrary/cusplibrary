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


#include <cusp/convert.h>
#include <cusp/multiply.h>

#include <cusp/detail/type_traits.h>

#include <thrust/execution_policy.h>

namespace cusp
{
namespace system
{
namespace cuda
{

template <typename DerivedPolicy,
         typename MatrixType,
         typename ArrayType1,
         typename ArrayType2>
void multiply_(cuda::execution_policy<DerivedPolicy>& exec,
              MatrixType& A,
              ArrayType1& x,
              ArrayType2& y,
              array2d_format,
              array1d_format,
              array1d_format)
{
    typedef typename cusp::detail::as_array2d_type<Matrix1,cusp::host_memory>::type Array2d;
    typedef typename ArrayType1::value_type ValueType1;
    typedef typename ArrayType2::value_type ValueType2;

    Array2d A_(A);
    cusp::array1d<ValueType1,cusp::host_memory> x_(x);
    cusp::array1d<ValueType2,cusp::host_memory> y_;

    cusp::multiply(A_,x_,y_);

    cusp::copy(y_, y);
}

template <typename DerivedPolicy,
         typename Matrix1,
         typename Matrix2,
         typename Matrix3>
void multiply_(cuda::execution_policy<DerivedPolicy>& exec,
              Matrix1& A,
              Matrix2& B,
              Matrix3& C,
              array2d_format,
              array2d_format,
              array2d_format)
{
    typedef typename cusp::detail::as_array2d_type<Matrix1,cusp::host_memory>::type Array2dMatrix1;
    typedef typename cusp::detail::as_array2d_type<Matrix2,cusp::host_memory>::type Array2dMatrix2;
    typedef typename cusp::detail::as_array2d_type<Matrix3,cusp::host_memory>::type Array2dMatrix3;

    Array2dMatrix1 A_(A);
    Array2dMatrix2 B_(B);
    Array2dMatrix3 C_;

    cusp::multiply(A_,B_,C_);

    cusp::convert(C_, C);
}

} // end namespace cuda
} // end namespace system
} // end namespace cusp
