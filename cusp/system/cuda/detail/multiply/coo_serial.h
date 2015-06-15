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

#include <thrust/device_ptr.h>

namespace cusp
{
namespace system
{
namespace cuda
{

// COO format SpMV kernel that uses only one thread
// This is incredibly slow, so it is only useful for testing purposes,
// *extremely* small matrices, or a few elements at the end of a
// larger matrix

template <typename IndexType, typename ValueType, typename BinaryFunction1, typename BinaryFunction2>
__global__ void
spmv_coo_serial_kernel(const IndexType num_entries,
                       const IndexType * I,
                       const IndexType * J,
                       const ValueType * V,
                       const ValueType * x,
                       ValueType * y,
                       BinaryFunction1 combine,
                       BinaryFunction2 reduce)
{
    for(IndexType n = 0; n < num_entries; n++)
    {
        y[I[n]] = reduce(y[I[n]], combine(V[n], x[J[n]]));
    }
}


template <typename Matrix,
          typename Array1,
          typename Array2,
          typename BinaryFunction1,
          typename BinaryFunction2>
void spmv_coo_serial_device(const Matrix& A,
                            const Array1& x,
                                  Array2& y,
                            BinaryFunction1 combine,
                            BinaryFunction2 reduce)
{
    typedef typename Matrix::index_type IndexType;
    typedef typename Matrix::value_type ValueType;

    const IndexType * I = thrust::raw_pointer_cast(&A.row_indices[0]);
    const IndexType * J = thrust::raw_pointer_cast(&A.column_indices[0]);
    const ValueType * V = thrust::raw_pointer_cast(&A.values[0]);

    const ValueType * x_ptr = thrust::raw_pointer_cast(&x[0]);
    ValueType * y_ptr = thrust::raw_pointer_cast(&y[0]);

    spmv_coo_serial_kernel<IndexType,ValueType,BinaryFunction1,BinaryFunction2> <<<1,1>>>
    (A.num_entries, I, J, V, x_ptr, y_ptr, combine, reduce);
}

} // end namespace cuda
} // end namespace system
} // end namespace cusp

