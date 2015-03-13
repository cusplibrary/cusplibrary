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

#include <cusp/sort.h>

#include <cusp/system/detail/generic/conversions/ell_to_other.h>

#include <thrust/copy.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy, typename SourceType, typename DestinationType>
typename enable_if_same_system<SourceType,DestinationType>::type
convert(thrust::execution_policy<DerivedPolicy>& exec,
        const SourceType& src,
        DestinationType& dst,
        cusp::hyb_format&,
        cusp::coo_format&)
{
    typedef typename SourceType::coo_matrix_type  CooMatrixType;
    typedef typename CooMatrixType::container  CooMatrix;

    // convert ell portion to coo
    CooMatrix temp;
    cusp::convert(exec, src.ell, temp);

    // resize output
    dst.resize(src.num_rows, src.num_cols, temp.num_entries + src.coo.num_entries);

    if(src.num_entries == 0) return;

    // merge coo matrices together
    thrust::copy(exec, temp.row_indices.begin(),       temp.row_indices.end(),       dst.row_indices.begin());
    thrust::copy(exec, temp.column_indices.begin(),    temp.column_indices.end(),    dst.column_indices.begin());
    thrust::copy(exec, temp.values.begin(),            temp.values.end(),            dst.values.begin());
    thrust::copy(exec, src.coo.row_indices.begin(),    src.coo.row_indices.end(),    dst.row_indices.begin()    + temp.num_entries);
    thrust::copy(exec, src.coo.column_indices.begin(), src.coo.column_indices.end(), dst.column_indices.begin() + temp.num_entries);
    thrust::copy(exec, src.coo.values.begin(),         src.coo.values.end(),         dst.values.begin()         + temp.num_entries);

    if (temp.num_entries > 0 && src.coo.num_entries > 0)
        cusp::sort_by_row_and_column(exec, dst.row_indices, dst.column_indices, dst.values);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp
