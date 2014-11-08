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

#include <cusp/copy.h>
#include <cusp/format.h>
#include <cusp/array1d.h>

#include <thrust/fill.h>
#include <thrust/extrema.h>
#include <thrust/binary_search.h>
#include <thrust/transform.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

namespace cusp
{

template <typename ArrayType1, typename ArrayType2, typename ArrayType3>
void sort_by_row(ArrayType1& row_indices, ArrayType2& column_indices, ArrayType3& values)
{
    typedef typename ArrayType1::value_type IndexType;
    typedef typename ArrayType3::value_type ValueType;
    typedef typename ArrayType1::memory_space MemorySpace;

    size_t N = row_indices.size();

    cusp::array1d<IndexType,MemorySpace> permutation(N);
    thrust::sequence(permutation.begin(), permutation.end());

    // compute permutation that sorts the row_indices
    thrust::sort_by_key(row_indices.begin(), row_indices.end(), permutation.begin());

    // copy column_indices and values to temporary buffers
    cusp::array1d<IndexType,MemorySpace> temp1(column_indices);
    cusp::array1d<ValueType,MemorySpace> temp2(values);

    // use permutation to reorder the values
    thrust::gather(permutation.begin(), permutation.end(),
                   thrust::make_zip_iterator(thrust::make_tuple(temp1.begin(),   temp2.begin())),
                   thrust::make_zip_iterator(thrust::make_tuple(column_indices.begin(), values.begin())));
}

template <typename ArrayType1, typename ArrayType2, typename ArrayType3>
void sort_by_row_and_column(ArrayType1& row_indices, ArrayType2& column_indices, ArrayType3& values)
{
    typedef typename ArrayType1::value_type IndexType;
    typedef typename ArrayType3::value_type ValueType;
    typedef typename ArrayType1::memory_space MemorySpace;

    size_t N = row_indices.size();

    cusp::array1d<IndexType,MemorySpace> permutation(N);
    thrust::sequence(permutation.begin(), permutation.end());

    // compute permutation and sort by (I,J)
    {
        cusp::array1d<IndexType,MemorySpace> temp(column_indices);
        thrust::stable_sort_by_key(temp.begin(), temp.end(), permutation.begin());

        cusp::copy(row_indices, temp);
        thrust::gather(permutation.begin(), permutation.end(), temp.begin(), row_indices.begin());
        thrust::stable_sort_by_key(row_indices.begin(), row_indices.end(), permutation.begin());

        cusp::copy(column_indices, temp);
        thrust::gather(permutation.begin(), permutation.end(), temp.begin(), column_indices.begin());
    }

    // use permutation to reorder the values
    {
        cusp::array1d<ValueType,MemorySpace> temp(values);
        thrust::gather(permutation.begin(), permutation.end(), temp.begin(), values.begin());
    }
}

} // end namespace cusp
