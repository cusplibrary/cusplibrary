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

#include <cusp/detail/format.h>
#include <cusp/array1d.h>
#include <cusp/system/detail/adl/sort.h>
#include <cusp/system/detail/generic/sort.h>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

namespace cusp
{

template <typename DerivedPolicy, typename ArrayType>
void counting_sort(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                   ArrayType& v, typename ArrayType::value_type min, typename ArrayType::value_type max)
{
    using cusp::system::detail::generic::counting_sort;

    counting_sort(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), v, min, max);
}

template <typename ArrayType>
void counting_sort(ArrayType& v, typename ArrayType::value_type min, typename ArrayType::value_type max)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType::memory_space System;

    System system;

    cusp::counting_sort(select_system(system), v, min, max);
}

template <typename DerivedPolicy, typename ArrayType1, typename ArrayType2>
void counting_sort_by_key(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                          ArrayType1& keys, ArrayType2& vals,
                          typename ArrayType1::value_type min, typename ArrayType1::value_type max)
{
    using cusp::system::detail::generic::counting_sort_by_key;

    counting_sort_by_key(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), keys, vals, min, max);
}

template <typename ArrayType1, typename ArrayType2>
void counting_sort_by_key(ArrayType1& keys, ArrayType2& vals,
                          typename ArrayType1::value_type min, typename ArrayType1::value_type max)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType1::memory_space System1;
    typedef typename ArrayType2::memory_space System2;

    System1 system1;
    System2 system2;

    cusp::counting_sort_by_key(select_system(system1,system2), keys, vals, min, max);
}

template <typename DerivedPolicy, typename ArrayType1, typename ArrayType2, typename ArrayType3>
void sort_by_row(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                 ArrayType1& row_indices, ArrayType2& column_indices, ArrayType3& values,
                 const int min_row, const int max_row)
{
    typedef typename ArrayType1::value_type IndexType;
    typedef typename ArrayType3::value_type ValueType;
    typedef typename ArrayType1::memory_space MemorySpace;

    size_t N = row_indices.size();

    IndexType minr = min_row == -1 ? 0 : min_row;
    IndexType maxr = max_row;

    if(max_row == -1)
      maxr = *thrust::max_element(row_indices.begin(), row_indices.end());

    cusp::array1d<IndexType,MemorySpace> permutation(N);
    thrust::sequence(exec, permutation.begin(), permutation.end());

    // compute permutation that sorts the row_indices
    IndexType max = *thrust::max_element(row_indices.begin(), row_indices.end());
    cusp::counting_sort_by_key(exec, row_indices, permutation, minr, maxr);

    // copy column_indices and values to temporary buffers
    cusp::array1d<IndexType,MemorySpace> temp1(column_indices);
    cusp::array1d<ValueType,MemorySpace> temp2(values);

    // use permutation to reorder the values
    thrust::gather(exec,
                   permutation.begin(), permutation.end(),
                   thrust::make_zip_iterator(thrust::make_tuple(temp1.begin(),   temp2.begin())),
                   thrust::make_zip_iterator(thrust::make_tuple(column_indices.begin(), values.begin())));
}

template <typename ArrayType1, typename ArrayType2, typename ArrayType3>
void sort_by_row(ArrayType1& row_indices, ArrayType2& column_indices, ArrayType3& values,
                 const int min_row, const int max_row)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType1::memory_space System1;
    typedef typename ArrayType2::memory_space System2;
    typedef typename ArrayType3::memory_space System3;

    System1 system1;
    System2 system2;
    System3 system3;

    cusp::sort_by_row(select_system(system1,system2,system3),
                      row_indices, column_indices, values,
                      min_row, max_row);
}

template <typename DerivedPolicy, typename ArrayType1, typename ArrayType2, typename ArrayType3>
void sort_by_row_and_column(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                            ArrayType1& row_indices, ArrayType2& column_indices, ArrayType3& values,
                            const int min_row, const int max_row, const int min_col, const int max_col)
{
    typedef typename ArrayType1::value_type IndexType;
    typedef typename ArrayType3::value_type ValueType;
    typedef typename ArrayType1::memory_space MemorySpace;

    size_t N = row_indices.size();

    cusp::array1d<IndexType,MemorySpace> permutation(N);
    thrust::sequence(exec, permutation.begin(), permutation.end());

    IndexType minr = min_row == -1 ? 0 : min_row;
    IndexType maxr = max_row;
    IndexType minc = min_col == -1 ? 0 : min_col;
    IndexType maxc = max_col;

    if(max_row == -1)
      maxr = *thrust::max_element(row_indices.begin(), row_indices.end());
    if(max_col == -1)
      maxc = *thrust::max_element(column_indices.begin(), column_indices.end());

    // compute permutation and sort by (I,J)
    {
        cusp::array1d<IndexType,MemorySpace> temp(column_indices);
        cusp::counting_sort_by_key(exec, temp, permutation, minc, maxc);

        thrust::copy(exec, row_indices.begin(), row_indices.end(), temp.begin());
        thrust::gather(exec, permutation.begin(), permutation.end(), temp.begin(), row_indices.begin());
        cusp::counting_sort_by_key(exec, row_indices, permutation, minr, maxr);

        thrust::copy(exec, column_indices.begin(), column_indices.end(), temp.begin());
        thrust::gather(exec, permutation.begin(), permutation.end(), temp.begin(), column_indices.begin());
    }

    // use permutation to reorder the values
    {
        cusp::array1d<ValueType,MemorySpace> temp(values);
        thrust::gather(exec, permutation.begin(), permutation.end(), temp.begin(), values.begin());
    }
}

template <typename ArrayType1, typename ArrayType2, typename ArrayType3>
void sort_by_row_and_column(ArrayType1& row_indices, ArrayType2& column_indices, ArrayType3& values,
                            const int min_row, const int max_row, const int min_col, const int max_col)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType1::memory_space System1;
    typedef typename ArrayType2::memory_space System2;
    typedef typename ArrayType3::memory_space System3;

    System1 system1;
    System2 system2;
    System3 system3;

    cusp::sort_by_row_and_column(select_system(system1,system2,system3),
                                 row_indices, column_indices, values,
                                 min_row, max_row, min_col, max_col);
}

} // end namespace cusp
