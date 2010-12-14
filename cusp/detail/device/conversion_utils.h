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

#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>

#include <cusp/detail/format_utils.h>

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <thrust/iterator/constant_iterator.h>


namespace cusp
{
namespace detail
{
namespace device
{
namespace detail
{

template <typename IndexType>
struct occupied_diagonal_functor
{
  typedef IndexType result_type;

  const   IndexType num_rows;

  occupied_diagonal_functor(const IndexType num_rows)
    : num_rows(num_rows) {}

  template <typename Tuple>
    __host__ __device__
  IndexType operator()(const Tuple& t) const
  {
    const IndexType i = thrust::get<0>(t);
    const IndexType j = thrust::get<1>(t);

    return j-i+num_rows;
  }
};

template <typename Matrix>
size_t count_diagonals(const Matrix& coo, cusp::coo_format)
{
    typedef typename Matrix::index_type IndexType;

    cusp::array1d<IndexType,cusp::device_memory> values(coo.num_rows+coo.num_cols,IndexType(0));

    thrust::scatter(thrust::constant_iterator<IndexType>(1), 
		    thrust::constant_iterator<IndexType>(1)+coo.num_entries, 
		    thrust::make_transform_iterator(thrust::make_zip_iterator( thrust::make_tuple( coo.row_indices.begin(), coo.column_indices.begin() ) ), 
						    occupied_diagonal_functor<IndexType>(coo.num_rows)), 
		    values.begin());

    return thrust::reduce(values.begin(), values.end());
}

template <typename Matrix>
size_t count_diagonals(const Matrix& csr, cusp::csr_format)
{
    typedef typename Matrix::index_type IndexType;

    // expand row offsets into row indices
    cusp::array1d<IndexType, cusp::device_memory> row_indices(csr.num_entries);
    cusp::detail::offsets_to_indices(csr.row_offsets, row_indices);

    cusp::array1d<IndexType,cusp::device_memory> values(csr.num_rows+csr.num_cols,IndexType(0));

    thrust::scatter(thrust::constant_iterator<IndexType>(1), 
		    thrust::constant_iterator<IndexType>(1)+csr.num_entries, 
		    thrust::make_transform_iterator(thrust::make_zip_iterator( thrust::make_tuple( row_indices.begin(), csr.column_indices.begin() ) ), 
						    occupied_diagonal_functor<IndexType>(csr.num_rows)), 
		    values.begin());

    return thrust::reduce(values.begin(), values.end());
}

template <typename Matrix>
size_t compute_max_entries_per_row(const Matrix& coo, cusp::coo_format)
{
    typedef typename Matrix::index_type IndexType;

    // expand row offsets into row indices
    cusp::array1d<IndexType, cusp::device_memory> row_offsets(coo.num_rows+1);
    cusp::detail::indices_to_offsets(coo.row_indices, row_offsets);

    size_t max_entries_per_row = 
    thrust::inner_product(row_offsets.begin() + 1, row_offsets.end(),
        		  row_offsets.begin(),
        		  IndexType(0),
        		  thrust::maximum<IndexType>(),
        		  thrust::minus<IndexType>());

    return max_entries_per_row;
}

template <typename Matrix>
size_t compute_max_entries_per_row(const Matrix& csr, cusp::csr_format)
{
    typedef typename Matrix::index_type IndexType;

    size_t max_entries_per_row = 
    thrust::inner_product(csr.row_offsets.begin() + 1, csr.row_offsets.end(),
        csr.row_offsets.begin(),
        IndexType(0),
        thrust::maximum<IndexType>(),
        thrust::minus<IndexType>());

    return max_entries_per_row;
}

template <typename Matrix>
size_t compute_optimal_entries_per_row(const Matrix& csr,
                                      float relative_speed,
                                      size_t breakeven_threshold,
                                      cusp::csr_format)
{
    typedef typename Matrix::index_type IndexType;
    
    // compute maximum row length
    IndexType max_cols_per_row = compute_max_entries_per_row(csr, cusp::csr_format());

    // allocate storage for the cumulative histogram and histogram
    cusp::array1d<IndexType,cusp::device_memory> histogram(max_cols_per_row + 1, IndexType(0));
    cusp::array1d<IndexType,cusp::device_memory> cumulative_histogram(max_cols_per_row + 1, IndexType(0));

    // compute distribution of nnz per row
    cusp::array1d<IndexType,cusp::device_memory> entries_per_row(csr.num_rows);
    thrust::adjacent_difference( csr.row_offsets.begin()+1, csr.row_offsets.end(), entries_per_row.begin() );

    // sort data to bring equal elements together
    thrust::sort(entries_per_row.begin(), entries_per_row.end());

    // find the end of each bin of values
    thrust::counting_iterator<IndexType> search_begin(0);
    thrust::upper_bound(entries_per_row.begin(),
                        entries_per_row.end(),
                        search_begin,
                        search_begin + max_cols_per_row + 1,
                        cumulative_histogram.begin());

    // compute the histogram by taking differences of the cumulative histogram
    thrust::adjacent_difference(cumulative_histogram.begin(),
                                cumulative_histogram.end(),
                                histogram.begin());

    // transfer to host 
    cusp::array1d<IndexType,cusp::host_memory> histogram_h(histogram);

    // compute optimal ELL column size 
    IndexType num_cols_per_row = max_cols_per_row;
    for(IndexType i = 0, rows = csr.num_rows; i < max_cols_per_row; i++)
    {
        rows -= histogram_h[i];  //number of rows of length > i
        if(relative_speed * rows < csr.num_rows || (size_t) rows < breakeven_threshold)
        {
            num_cols_per_row = i;
            break;
        }
    }

    return num_cols_per_row;
}

} // end namespace detail

template <typename Matrix>
size_t count_diagonals(const Matrix& m)
{
  return cusp::detail::device::detail::count_diagonals(m, typename Matrix::format());
}

template <typename Matrix>
size_t compute_max_entries_per_row(const Matrix& m)
{
  return cusp::detail::device::detail::compute_max_entries_per_row(m, typename Matrix::format());
}


////////////////////////////////////////////////////////////////////////////////
//! Compute Optimal Number of Columns per Row in the ELL part of the HYB format
//! Examines the distribution of nonzeros per row of the input CSR matrix to find
//! the optimal tradeoff between the ELL and COO portions of the hybrid (HYB)
//! sparse matrix format under the assumption that ELL performance is a fixed
//! multiple of COO performance.  Furthermore, since ELL performance is also
//! sensitive to the absolute number of rows (and COO is not), a threshold is
//! used to ensure that the ELL portion contains enough rows to be worthwhile.
//! The default values were chosen empirically for a GTX280.
//!
//! @param csr                  CSR matrix
//! @param relative_speed       Speed of ELL relative to COO (e.g. 2.0 -> ELL is twice as fast)
//! @param breakeven_threshold  Minimum threshold at which ELL is faster than COO
////////////////////////////////////////////////////////////////////////////////
template <typename Matrix>
size_t compute_optimal_entries_per_row(const Matrix& m,
                                       float relative_speed = 3.0f,
                                       size_t breakeven_threshold = 4096)
{
  return cusp::detail::device::detail::compute_optimal_entries_per_row
    (m, relative_speed, breakeven_threshold, typename Matrix::format());
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

