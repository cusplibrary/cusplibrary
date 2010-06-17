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
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>
#include <cusp/array2d.h>

#include <cusp/detail/format_utils.h>

//#include <cusp/detail/device/conversion.h>
//#include <cusp/detail/host/conversion_utils.h>

#include <cusp/detail/host/convert.h>

#include <algorithm>
#include <string>
#include <stdexcept>

namespace cusp
{
namespace detail
{
namespace device
{

// Device Conversion Functions
// COO <- CSR
//     <- Array
// CSR <- COO
//     <- DIA
//     <- ELL
//     <- HYB
//     <- Array
// DIA <- CSR
// ELL <- CSR
// HYB <- CSR
// Array <- COO
//       <- CSR
//       <- Array (different Orientation)

/////////
// COO //
/////////
template <typename IndexType, typename ValueType>
void convert(const cusp::csr_matrix<IndexType,ValueType,cusp::device_memory>& src,
                   cusp::coo_matrix<IndexType,ValueType,cusp::device_memory>& dst)
             
{
    cusp::coo_matrix<IndexType,ValueType,cusp::device_memory> tmp(src.num_rows, src.num_cols, src.num_entries);

    cusp::detail::offsets_to_indices(src.row_offsets, tmp.row_indices);
    tmp.column_indices = src.column_indices;
    tmp.values = src.values;

    dst.swap(tmp);
}

/////////
// CSR //
/////////
template <typename IndexType, typename ValueType>
void convert(const cusp::coo_matrix<IndexType,ValueType,cusp::device_memory>& src,
                   cusp::csr_matrix<IndexType,ValueType,cusp::device_memory>& dst)
{
    cusp::csr_matrix<IndexType,ValueType,cusp::device_memory> tmp(src.num_rows, src.num_cols, src.num_entries);

    cusp::detail::indices_to_offsets(src.row_indices, tmp.row_offsets);
    tmp.column_indices = src.column_indices;
    tmp.values         = src.values;

    dst.swap(tmp);
}

/////////
// DIA //
/////////

/////////
// ELL //
/////////

/////////
// HYB //
/////////

///////////
// Array //
///////////

///////////////////
// Host Fallback //
///////////////////
template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst)
{
    // transfer to host, convert on host, and transfer back to device
    typedef typename Matrix1::template rebind<cusp::host_memory>::type      HostSourceType;
    typedef typename Matrix2::template rebind<cusp::host_memory>::type HostDestinationType;

    HostSourceType tmp1(src);

    HostDestinationType tmp2;

    cusp::detail::host::convert(tmp1, tmp2);

    dst = tmp2;
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

