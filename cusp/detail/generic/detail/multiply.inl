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

#include <cusp/coo_matrix.h>
#include <cusp/array2d.h>

#include <cusp/detail/format_utils.h>

#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/segmented_scan.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/iterator/permutation_iterator.h>

namespace cusp
{
namespace detail
{
namespace generic
{

template <typename IndexType,
          typename ValueType,
          typename MemorySpace>
void multiply(const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& A,
              const cusp::coo_matrix<IndexType,ValueType,MemorySpace>& B,
                    cusp::coo_matrix<IndexType,ValueType,MemorySpace>& C)
{
    // check whether matrices are empty
    if (A.num_entries == 0 || B.num_entries == 0)
    {
        cusp::coo_matrix<IndexType,ValueType,MemorySpace> temp(A.num_rows, B.num_cols, 0);
        C.swap(temp);
        return;
    }

    // compute row offsets for B
    cusp::array1d<IndexType,MemorySpace> B_row_offsets(B.num_rows + 1);
    cusp::detail::indices_to_offsets(B.row_indices, B_row_offsets);

    // compute row lengths for B
    cusp::array1d<IndexType,MemorySpace> B_row_lengths(B.num_rows);
    thrust::transform(B_row_offsets.begin() + 1, B_row_offsets.end(), B_row_offsets.begin(), B_row_lengths.begin(), thrust::minus<IndexType>());

    // for each element A(i,j) compute the number of nonzero elements in B(j,:)
    cusp::array1d<IndexType,MemorySpace> segment_lengths(A.num_entries);
    thrust::next::gather(A.column_indices.begin(), A.column_indices.end(),
                   B_row_lengths.begin(),
                   segment_lengths.begin());
    
    // output pointer
    cusp::array1d<IndexType,MemorySpace> output_ptr(A.num_entries + 1);
    thrust::exclusive_scan(segment_lengths.begin(), segment_lengths.end(),
                           output_ptr.begin(),
                           IndexType(0));
    output_ptr[A.num_entries] = output_ptr[A.num_entries - 1] + segment_lengths[A.num_entries - 1]; // XXX is this necessary?

    IndexType coo_num_nonzeros = output_ptr[A.num_entries];
    
    // enumerate the segments in the intermediate format corresponding to each entry A(i,j)
    // XXX could be done with offset_to_index instead
    cusp::array1d<IndexType,MemorySpace> segments(coo_num_nonzeros, 0);
    thrust::scatter_if(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(A.num_entries),
                       output_ptr.begin(), 
                       segment_lengths.begin(),
                       segments.begin());
    thrust::inclusive_scan(segments.begin(), segments.end(), segments.begin(), thrust::maximum<IndexType>());
   
    // compute gather locations of intermediate format
    cusp::array1d<IndexType,MemorySpace> gather_locations(coo_num_nonzeros, 1);
    thrust::scatter_if(thrust::make_permutation_iterator(B_row_offsets.begin(), A.column_indices.begin()),
                       thrust::make_permutation_iterator(B_row_offsets.begin(), A.column_indices.begin()) + A.num_entries,
                       output_ptr.begin(),
                       segment_lengths.begin(),
                       gather_locations.begin());
    thrust::experimental::inclusive_segmented_scan(gather_locations.begin(), gather_locations.end(),
                                                   segments.begin(),
                                                   gather_locations.begin());
    
    // compute column entries and values of intermediate format
    cusp::array1d<IndexType,MemorySpace> I(coo_num_nonzeros);
    cusp::array1d<IndexType,MemorySpace> J(coo_num_nonzeros);
    cusp::array1d<ValueType,MemorySpace> V(coo_num_nonzeros);
    
    thrust::next::gather(segments.begin(), segments.end(),
                         A.row_indices.begin(),
                         I.begin());

    thrust::next::gather(gather_locations.begin(), gather_locations.end(),
                         B.column_indices.begin(),
                         J.begin());

    thrust::transform(thrust::make_permutation_iterator(A.values.begin(), segments.begin()),
                      thrust::make_permutation_iterator(A.values.begin(), segments.begin()) + coo_num_nonzeros,
                      thrust::make_permutation_iterator(B.values.begin(), gather_locations.begin()),
                      V.begin(),
                      thrust::multiplies<ValueType>());

    // sort by (I,J)
    {
        // TODO use explicit permuation and temporary arrays for efficiency
        thrust::sort_by_key(J.begin(), J.end(), thrust::make_zip_iterator(thrust::make_tuple(I.begin(), V.begin())));
        thrust::sort_by_key(I.begin(), I.end(), thrust::make_zip_iterator(thrust::make_tuple(J.begin(), V.begin())));
    }


    // compute unique number of nonzeros in the output
    IndexType NNZ = thrust::inner_product(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
                                          thrust::make_zip_iterator(thrust::make_tuple(I.end (), J.end()))    - 1,
                                          thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())) + 1,
                                          IndexType(0),
                                          thrust::plus<IndexType>(),
                                          thrust::not_equal_to< thrust::tuple<IndexType,IndexType> >()) + 1;

    // allocate space for output
    C.resize(A.num_rows, B.num_cols, NNZ);

    // sum values with the same (i,j)
    thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(I.begin(), J.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(I.end(),   J.end())),
                          V.begin(),
                          thrust::make_zip_iterator(thrust::make_tuple(C.row_indices.begin(), C.column_indices.begin())),
                          C.values.begin(),
                          thrust::equal_to< thrust::tuple<IndexType,IndexType> >(),
                          thrust::plus<ValueType>());
}


template <typename ValueType,
          typename MemorySpace>
void multiply(const cusp::array2d<ValueType,MemorySpace>& A,
              const cusp::array2d<ValueType,MemorySpace>& B,
                    cusp::array2d<ValueType,MemorySpace>& C)
{
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

} // end namespace generic
} // end namespace detail
} // end namespace cusp

