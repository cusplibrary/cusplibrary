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

#include <cusp/format.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/blas.h>

#include <cusp/detail/format_utils.h>
#include <cusp/detail/host/convert.h>

#include <thrust/gather.h>
#include <thrust/inner_product.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

namespace cusp
{
namespace detail
{
namespace device
{

// Device Conversion Functions
// COO <- CSR
// COO <- ELL
// CSR <- COO
// CSR <- ELL
// ELL <- CSR
// ELL <- COO
// 
// All other conversions happen on the host

struct is_nonnegative
{
  template<typename T>
  __host__ __device__
  bool operator()(T &x)
  {
    return x > T(-1);
  } // end operator()()
}; // end is_nonnegative


/////////
// COO //
/////////
template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::csr_format,
             cusp::coo_format)
{
    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    cusp::detail::offsets_to_indices(src.row_offsets, dst.row_indices);
    cusp::copy(src.column_indices, dst.column_indices);
    cusp::copy(src.values,         dst.values);
}


template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::ell_format,
             cusp::coo_format)
{
   typedef typename Matrix2::index_type IndexType;

   // allocate output storage
   IndexType num_entries = src.column_indices.values.size();
   dst.resize(src.num_rows, src.num_cols, num_entries);

   // form row indices 
   IndexType num_entries_per_row = src.column_indices.num_cols;
   cusp::array1d<IndexType, cusp::device_memory> row_offsets(src.num_rows+1);
   thrust::sequence( row_offsets.begin(), row_offsets.end(), IndexType(0), num_entries_per_row ); 

   // expand row offsets into row indices
   cusp::detail::offsets_to_indices(row_offsets, dst.row_indices);

   // compute column-major to row-major permutation
   cusp::array1d<IndexType, cusp::device_memory> permutation(num_entries);
   thrust::transform(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(0)+num_entries, 
		     thrust::constant_iterator<IndexType>(num_entries_per_row), 
		     permutation.begin(), 
		     thrust::modulus<IndexType>());

   // scale by pitch and add row index
   cusp::blas::axpby(permutation, dst.row_indices,
                     permutation,
                     IndexType(src.column_indices.pitch),
                     IndexType(1));

   // store src.column_indices and src.values in row-major format in dst
   thrust::gather(permutation.begin(), permutation.end(),
		  thrust::make_zip_iterator(thrust::make_tuple(	src.column_indices.values.begin(), 
					     			src.values.values.begin())),
		  thrust::make_zip_iterator(thrust::make_tuple( dst.column_indices.begin(),
					     			dst.values.begin())));

   // copy only valid ELL entries to COO
   thrust::copy_if(thrust::make_zip_iterator(thrust::make_tuple(dst.row_indices.begin(), 
					     			dst.column_indices.begin(), 
					     			dst.values.begin())),
		   thrust::make_zip_iterator(thrust::make_tuple(dst.row_indices.end(),
					     			dst.column_indices.end(),
					     			dst.values.end())),
                   dst.column_indices.begin(),
		   thrust::make_zip_iterator(thrust::make_tuple(dst.row_indices.begin(),
					     			dst.column_indices.begin(),
					     			dst.values.begin())),
       	    	   is_nonnegative());

   // resize dst to remove discarded entries
   dst.resize(dst.num_rows, dst.num_cols, src.num_entries);
}


/////////
// CSR //
/////////
template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::coo_format,
             cusp::csr_format)
{
    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    cusp::detail::indices_to_offsets(src.row_indices, dst.row_offsets);
    cusp::copy(src.column_indices, dst.column_indices);
    cusp::copy(src.values,         dst.values);
}

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::ell_format,
             cusp::csr_format)
{
   typedef typename Matrix2::index_type IndexType;

   // allocate output storage
   IndexType num_entries = src.column_indices.values.size();
   dst.resize(src.num_rows, src.num_cols, num_entries);

   // form row indices 
   IndexType num_entries_per_row = src.column_indices.num_cols;
   cusp::array1d<IndexType, cusp::device_memory> row_offsets(src.num_rows+1);
   thrust::sequence( row_offsets.begin(), row_offsets.end(), IndexType(0), num_entries_per_row ); 

   // expand row offsets into row indices
   cusp::array1d<IndexType, cusp::device_memory> row_indices(num_entries);
   cusp::detail::offsets_to_indices(row_offsets, row_indices);

   // compute column-major to row-major permutation
   cusp::array1d<IndexType, cusp::device_memory> permutation(num_entries);
   thrust::transform(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(0)+num_entries, 
		     thrust::constant_iterator<IndexType>(num_entries_per_row), 
		     permutation.begin(), 
		     thrust::modulus<IndexType>());

   // scale by pitch and add row index
   cusp::blas::axpby(permutation, row_indices,
                     permutation,
                     IndexType(src.column_indices.pitch),
                     IndexType(1));

   // store src.column_indices and src.values in row-major format in dst
   thrust::gather(permutation.begin(), permutation.end(),
		  thrust::make_zip_iterator(thrust::make_tuple(	src.column_indices.values.begin(), 
					     			src.values.values.begin())),
		  thrust::make_zip_iterator(thrust::make_tuple( dst.column_indices.begin(),
					     			dst.values.begin())));

   // copy only valid ELL entries to CSR
   thrust::copy_if(thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), 
					     			dst.column_indices.begin(), 
					     			dst.values.begin())),
		   thrust::make_zip_iterator(thrust::make_tuple(row_indices.end(),
					     			dst.column_indices.end(),
					     			dst.values.end())),
                   dst.column_indices.begin(),
		   thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(),
					     			dst.column_indices.begin(),
					     			dst.values.begin())),
       	    	   is_nonnegative());

   // resize dst to remove discarded entries
   row_indices.resize(src.num_entries);
   dst.resize(dst.num_rows, dst.num_cols, src.num_entries);

   // contract row indices into row offsets
   cusp::detail::indices_to_offsets(row_indices, dst.row_offsets);
}

/////////
// DIA //
/////////

/////////
// ELL //
/////////
template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::coo_format,
             cusp::ell_format)
{
  typedef typename Matrix2::index_type IndexType;
  typedef typename Matrix2::value_type ValueType;

  // TODO centralize this somehow
  const size_t alignment = 32;

  // contract row indices into row offsets
  cusp::array1d<IndexType, cusp::device_memory> row_offsets(src.num_rows+1);
  cusp::detail::indices_to_offsets(src.row_indices, row_offsets);

  // compute maximum number of entries per row
  IndexType num_entries_per_row = 
    thrust::inner_product(row_offsets.begin() + 1, row_offsets.end(),
        row_offsets.begin(),
        IndexType(0),
        thrust::maximum<IndexType>(),
        thrust::minus<IndexType>());

  // allocate output storage
  dst.resize(src.num_rows, src.num_cols, src.num_entries, num_entries_per_row, alignment);

  // compute permutation from COO index to ELL index
  // first enumerate the entries within each row, e.g. [0, 1, 2, 0, 1, 2, 3, ...]
  cusp::array1d<IndexType, cusp::device_memory> permutation(src.num_entries);
  thrust::exclusive_scan_by_key(src.row_indices.begin(), src.row_indices.end(),
                                thrust::constant_iterator<IndexType>(1),
                                permutation.begin(),
                                IndexType(0));
  
  // next, scale by pitch and add row index
  cusp::blas::axpby(permutation, src.row_indices,
                    permutation,
                    IndexType(dst.column_indices.pitch),
                    IndexType(1));

  // fill output with padding
  thrust::fill(dst.column_indices.values.begin(), dst.column_indices.values.end(), IndexType(-1));
  thrust::fill(dst.values.values.begin(),         dst.values.values.end(),         ValueType(0));

  // scatter COO entries to ELL
  thrust::scatter(src.column_indices.begin(), src.column_indices.end(),
                  permutation.begin(),
                  dst.column_indices.values.begin());
  thrust::scatter(src.values.begin(), src.values.end(),
                  permutation.begin(),
                  dst.values.values.begin());
}

template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst,
             cusp::csr_format,
             cusp::ell_format)
{
  typedef typename Matrix2::index_type IndexType;
  typedef typename Matrix2::value_type ValueType;

  // TODO centralize this somehow
  const size_t alignment = 32;

  // compute maximum number of entries per row
  IndexType num_entries_per_row = 
    thrust::inner_product(src.row_offsets.begin() + 1, src.row_offsets.end(),
        src.row_offsets.begin(),
        IndexType(0),
        thrust::maximum<IndexType>(),
        thrust::minus<IndexType>());

  // allocate output storage
  dst.resize(src.num_rows, src.num_cols, src.num_entries, num_entries_per_row, alignment);

  // expand row offsets into row indices
  cusp::array1d<IndexType, cusp::device_memory> row_indices(src.num_entries);
  cusp::detail::offsets_to_indices(src.row_offsets, row_indices);

  // compute permutation from CSR index to ELL index
  // first enumerate the entries within each row, e.g. [0, 1, 2, 0, 1, 2, 3, ...]
  cusp::array1d<IndexType, cusp::device_memory> permutation(src.num_entries);
  thrust::exclusive_scan_by_key(row_indices.begin(), row_indices.end(),
                                thrust::constant_iterator<IndexType>(1),
                                permutation.begin(),
                                IndexType(0));
  
  // next, scale by pitch and add row index
  cusp::blas::axpby(permutation, row_indices,
                    permutation,
                    IndexType(dst.column_indices.pitch),
                    IndexType(1));

  // fill output with padding
  thrust::fill(dst.column_indices.values.begin(), dst.column_indices.values.end(), IndexType(-1));
  thrust::fill(dst.values.values.begin(),         dst.values.values.end(),         ValueType(0));

  // scatter CSR entries to ELL
  thrust::scatter(src.column_indices.begin(), src.column_indices.end(),
                  permutation.begin(),
                  dst.column_indices.values.begin());
  thrust::scatter(src.values.begin(), src.values.end(),
                  permutation.begin(),
                  dst.values.values.begin());
}


/////////
// HYB //
/////////

///////////
// Array //
///////////

///////////////////
// Host Fallback //
///////////////////
template <typename Matrix1, typename Matrix2, typename MatrixFormat1, typename MatrixFormat2>
void convert(const Matrix1& src, Matrix2& dst,
             MatrixFormat1,
             MatrixFormat2)
{
    // transfer to host, convert on host, and transfer back to device
    typedef typename Matrix1::container SourceContainerType;
    typedef typename Matrix2::container DestinationContainerType;
    typedef typename DestinationContainerType::template rebind<cusp::host_memory>::type HostDestinationContainerType;
    typedef typename SourceContainerType::template      rebind<cusp::host_memory>::type HostSourceContainerType;

    HostSourceContainerType tmp1(src);

    HostDestinationContainerType tmp2;

    cusp::detail::host::convert(tmp1, tmp2);

    cusp::copy(tmp2, dst);
}

/////////////////
// Entry Point //
/////////////////
template <typename Matrix1, typename Matrix2>
void convert(const Matrix1& src, Matrix2& dst)
{
    cusp::detail::device::convert(src, dst,
            typename Matrix1::format(),
            typename Matrix2::format());
}

} // end namespace device
} // end namespace detail
} // end namespace cusp

