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

#include <cusp/coo_matrix.h>

namespace cusp
{

namespace detail
{

namespace device
{

// COO format SpMV kernel that uses only one thread
// This is incredibly slow, so it is only useful for testing purposes,
// *extremely* small matrices, or a few elements at the end of a 
// larger matrix

template <typename IndexType, typename ValueType>
__global__ void
spmv_coo_serial_kernel(const IndexType num_nonzeros,
                       const IndexType * I, 
                       const IndexType * J, 
                       const ValueType * V, 
                       const ValueType * x, 
                             ValueType * y)
{
    for(IndexType n = 0; n < num_nonzeros; n++){
        y[I[n]] += V[n] * x[J[n]];
    }
}


template <typename IndexType, typename ValueType>
void spmv_coo_serial_device(const coo_matrix<IndexType,ValueType,cusp::device_memory>& d_coo, 
                            const ValueType * d_x, 
                                  ValueType * d_y)
{
    spmv_coo_serial_kernel<IndexType,ValueType> <<<1,1>>>
        (d_coo.num_nonzeros, d_coo.I, d_coo.J, d_coo.V, d_x, d_y);
}

} // end namespace device

} // end namespace detail

} // end namespace cusp

