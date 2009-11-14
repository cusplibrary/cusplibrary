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


#include <cusp/array1d.h>
#include <cusp/exception.h>

#include <cusp/detail/dispatch/spmv.h>

namespace cusp
{
namespace detail
{

template <typename MatrixType,
          typename T1, typename MemorySpace1,
          typename T2, typename MemorySpace2>
void assert_compatible_dimensions(const MatrixType& A,
                                  const cusp::array1d<T1, MemorySpace1>& x,
                                  const cusp::array1d<T2, MemorySpace2>& y)
{
    if(A.num_cols != x.size() || A.num_rows != y.size())
        throw cusp::invalid_input_exception("incompatible dimensions");
}

template <typename MatrixType,
          typename VectorType1,
          typename VectorType2>
void spmv(const MatrixType& A,
          const VectorType1& x,
                VectorType2& y)
{
    detail::assert_compatible_dimensions(A, x, y);

    cusp::detail::dispatch::spmv(A, x, y,
            typename MatrixType::memory_space(),
            typename VectorType1::memory_space(),
            typename VectorType2::memory_space());
}

} // end namespace detail
} // end namespace cusp

