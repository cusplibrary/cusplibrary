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

#include <cusp/blas.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/format.h>
#include <cusp/print.h>

#include <cusp/detail/format_utils.h>
#include <cusp/detail/device/conversion_utils.h>
#include <cusp/detail/host/convert.h>

#include <thrust/count.h>
#include <thrust/gather.h>
#include <thrust/inner_product.h>
#include <thrust/replace.h>
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
//     <- ELL
//     <- DIA
// CSR <- COO
//     <- ELL
//     <- DIA
// ELL <- CSR
//     <- COO
// DIA <- CSR
//     <- COO
// 
// All other conversions happen on the host

template <typename IndexType>
struct is_valid_ell_index
{
  const IndexType num_rows;

  is_valid_ell_index(const IndexType num_rows)
    : num_rows(num_rows) {}

  template <typename Tuple>
    __host__ __device__
  bool operator()(const Tuple& t) const
  {
    const IndexType i = thrust::get<0>(t);
    const IndexType j = thrust::get<1>(t);

    return i < num_rows && j != IndexType(-1);
  }
};

template <typename IndexType, typename ValueType>
struct is_valid_coo_index
{
  const IndexType num_rows;
  const IndexType num_cols;

  is_valid_coo_index(const IndexType num_rows, const IndexType num_cols)
    : num_rows(num_rows), num_cols(num_cols) {}

  template <typename Tuple>
    __host__ __device__
  bool operator()(const Tuple& t) const
  {
    const IndexType i = thrust::get<0>(t);
    const IndexType j = thrust::get<1>(t);
    const ValueType value = thrust::get<2>(t);

    return ( i > IndexType(-1) && i < num_rows ) && 
	   ( j > IndexType(-1) && j < num_cols ) && 
	   ( value != ValueType(0) ) ;
  }
};

template <typename T>
struct modulus_value : public thrust::unary_function<T,T>
{
  const T value;

  modulus_value(const T value)
    : value(value) {}

    __host__ __device__
  T operator()(const T& x) const
  {
    return x % value;
  }
};

template <typename T>
struct transpose_index_functor : public thrust::unary_function<T,T>
{
  const T num_entries_per_row;
  const T pitch;

  transpose_index_functor(const T pitch, const T num_entries_per_row)
    : num_entries_per_row(num_entries_per_row), pitch(pitch) {}

    __host__ __device__
  T operator()(const T& n) const
  {
    return pitch * (n % num_entries_per_row) + n / num_entries_per_row;
  }
};

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

    return num_rows-i+j;
  }
};

template <typename IndexType>
struct diagonal_index_functor : public thrust::unary_function<IndexType,IndexType>
{
  const IndexType pitch;

  diagonal_index_functor(const IndexType pitch)
    : pitch(pitch) {}

  template <typename Tuple>
    __host__ __device__
  IndexType operator()(const Tuple& t) const
  {
    const IndexType row  = thrust::get<0>(t);
    const IndexType diag = thrust::get<1>(t);

    return (diag*pitch) + row;
  }
};

template <typename IndexType>
struct is_positive
{
  __host__ __device__
  bool operator()(const IndexType x)
  {
    return x > 0;
  }
};

/////////
// COO //
/////////
template <typename Matrix1, typename Matrix2>
void csr_to_coo(const Matrix1& src, Matrix2& dst)
{
    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    cusp::detail::offsets_to_indices(src.row_offsets, dst.row_indices);
    cusp::copy(src.column_indices, dst.column_indices);
    cusp::copy(src.values,         dst.values);
}


template <typename Matrix1, typename Matrix2>
void ell_to_coo(const Matrix1& src, Matrix2& dst)
{
   typedef typename Matrix1::index_type IndexType;
   
   const IndexType pitch               = src.column_indices.pitch;
   const IndexType num_entries_per_row = src.column_indices.num_cols;

   // define types used to programatically generate row_indices
   typedef typename thrust::counting_iterator<IndexType> IndexIterator;
   typedef typename thrust::transform_iterator<modulus_value<IndexType>, IndexIterator> RowIndexIterator;

   RowIndexIterator row_indices_begin(IndexIterator(0), modulus_value<IndexType>(pitch));

   // compute true number of nonzeros in ELL
   const IndexType num_entries = 
     thrust::count_if
      (thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin())),
       thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin())) + src.column_indices.values.size(),
       is_valid_ell_index<IndexType>(src.num_rows));

   // allocate output storage
   dst.resize(src.num_rows, src.num_cols, num_entries);

   // copy valid entries to COO format
   thrust::copy_if
     (thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin(), src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_entries_per_row))),
      thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin(), src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_entries_per_row))) + src.column_indices.values.size(),
      thrust::make_zip_iterator(thrust::make_tuple(dst.row_indices.begin(), dst.column_indices.begin(), dst.values.begin())),
      is_valid_ell_index<IndexType>(src.num_rows));
}

template <typename Matrix1, typename Matrix2>
void dia_to_coo(const Matrix1& src, Matrix2& dst)
{
   typedef typename Matrix1::index_type IndexType;
   typedef typename Matrix1::value_type ValueType;
   
   // allocate output storage
   dst.resize(src.num_rows, src.num_cols, src.num_entries);

   if( src.num_entries == 0 ) return;

   const IndexType pitch = src.values.pitch;
   const size_t num_entries   = src.values.values.size();
   const size_t num_diagonals = src.diagonal_offsets.size();

   // define types used to programatically generate row_indices
   typedef typename thrust::counting_iterator<IndexType> IndexIterator;
   typedef typename thrust::transform_iterator<modulus_value<IndexType>, IndexIterator> RowIndexIterator;

   RowIndexIterator row_indices_begin(IndexIterator(0), modulus_value<IndexType>(pitch));

   cusp::array1d<IndexType, cusp::device_memory> scan_offsets(num_diagonals);
   thrust::sequence( scan_offsets.begin(), scan_offsets.end(), IndexType(0), IndexType(src.values.num_rows) ); 

   IndexType min = src.diagonal_offsets[0];
   // fill column entries with 1 minus the smallest diagonal, e.g. [-4, -4, -4, -4, -4, -4, -4, ...]
   cusp::array1d<IndexType, cusp::device_memory> column_indices(num_entries, min-1);
   // scatter the diagonal offsets to the first entry in each column, e.g. [-3, -4, -4, -1, -4, -4, 0, ...]
   thrust::scatter(src.diagonal_offsets.begin(),
		   src.diagonal_offsets.end(),
		   scan_offsets.begin(),
                   column_indices.begin());

   cusp::array1d<IndexType, cusp::device_memory> scan_row_indices(num_entries);
   cusp::detail::offsets_to_indices( scan_offsets, scan_row_indices );
   // enumerate the column entries, e.g. [-3, -3, -3, -1, -1, -1, 0, ...]
   thrust::inclusive_scan_by_key(scan_row_indices.begin(), 
				 scan_row_indices.end(),
			  	 column_indices.begin(), 
			  	 column_indices.begin(), 
				 thrust::equal_to<IndexType>(),
			  	 thrust::maximum<IndexType>());

   // enumerate the offsets within each column, e.g. [0, 1, 2, 0, 1, 2, 0, ...]
   cusp::array1d<IndexType, cusp::device_memory> column_offsets(num_entries);
   thrust::transform(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(num_entries), 
                     thrust::constant_iterator<IndexType>(pitch), 
                     column_offsets.begin(), 
                     thrust::modulus<IndexType>());

   // sum the indices and the offsets for the final column_indices array 
   cusp::blas::axpy( column_offsets, column_indices, IndexType(1) );

   // copy valid entries to COO format
   thrust::copy_if
     (thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices.begin(), src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_diagonals))),
      thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices.begin(), src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_diagonals))) + num_entries,
      thrust::make_zip_iterator(thrust::make_tuple(dst.row_indices.begin(), dst.column_indices.begin(), dst.values.begin())),
      is_valid_coo_index<IndexType,ValueType>(src.num_rows,src.num_cols));
}


/////////
// CSR //
/////////
template <typename Matrix1, typename Matrix2>
void coo_to_csr(const Matrix1& src, Matrix2& dst)
{
    dst.resize(src.num_rows, src.num_cols, src.num_entries);

    cusp::detail::indices_to_offsets(src.row_indices, dst.row_offsets);
    cusp::copy(src.column_indices, dst.column_indices);
    cusp::copy(src.values,         dst.values);
}

template <typename Matrix1, typename Matrix2>
void ell_to_csr(const Matrix1& src, Matrix2& dst)
{
   typedef typename Matrix1::index_type IndexType;
   
   const IndexType pitch               = src.column_indices.pitch;
   const IndexType num_entries_per_row = src.column_indices.num_cols;

   // define types used to programatically generate row_indices
   typedef typename thrust::counting_iterator<IndexType> IndexIterator;
   typedef typename thrust::transform_iterator<modulus_value<IndexType>, IndexIterator> RowIndexIterator;

   RowIndexIterator row_indices_begin(IndexIterator(0), modulus_value<IndexType>(pitch));

   // compute true number of nonzeros in ELL
   const IndexType num_entries = 
     thrust::count_if
      (thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin())),
       thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin())) + src.column_indices.values.size(),
       is_valid_ell_index<IndexType>(src.num_rows));

   // allocate output storage
   dst.resize(src.num_rows, src.num_cols, num_entries);

   // create temporary row_indices array to capture valid ELL row indices
   cusp::array1d<IndexType, cusp::device_memory> row_indices(num_entries);

   // copy valid entries to mixed COO/CSR format
   thrust::copy_if
     (thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin(), src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_entries_per_row))),
      thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, src.column_indices.values.begin(), src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_entries_per_row))) + src.column_indices.values.size(),
      thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), dst.column_indices.begin(), dst.values.begin())),
      is_valid_ell_index<IndexType>(src.num_rows));

   // convert COO row_indices to CSR row_offsets
   cusp::detail::indices_to_offsets(row_indices, dst.row_offsets);
}

template <typename Matrix1, typename Matrix2>
void dia_to_csr(const Matrix1& src, Matrix2& dst)
{
   typedef typename Matrix1::index_type IndexType;
   typedef typename Matrix1::value_type ValueType;
   
   // allocate output storage
   dst.resize(src.num_rows, src.num_cols, src.num_entries);

   if( src.num_entries == 0 ) return;

   const IndexType pitch = src.values.pitch;
   const size_t num_entries   = src.values.values.size();
   const size_t num_diagonals = src.diagonal_offsets.size();

   // define types used to programatically generate row_indices
   typedef typename thrust::counting_iterator<IndexType> IndexIterator;
   typedef typename thrust::transform_iterator<modulus_value<IndexType>, IndexIterator> RowIndexIterator;

   RowIndexIterator row_indices_begin(IndexIterator(0), modulus_value<IndexType>(pitch));

   cusp::array1d<IndexType, cusp::device_memory> scan_offsets(num_diagonals);
   thrust::sequence( scan_offsets.begin(), scan_offsets.end(), IndexType(0), IndexType(src.values.num_rows) ); 

   IndexType min = src.diagonal_offsets[0];
   // fill column entries with 1 minus the smallest diagonal, e.g. [-4, -4, -4, -4, -4, -4, -4, ...]
   cusp::array1d<IndexType, cusp::device_memory> column_indices(num_entries, min-1);
   // scatter the diagonal offsets to the first entry in each column, e.g. [-3, -4, -4, -1, -4, -4, 0, ...]
   thrust::scatter(src.diagonal_offsets.begin(),
		   src.diagonal_offsets.end(),
		   scan_offsets.begin(),
                   column_indices.begin());

   cusp::array1d<IndexType, cusp::device_memory> scan_row_indices(num_entries);
   cusp::detail::offsets_to_indices( scan_offsets, scan_row_indices );
   // enumerate the column entries, e.g. [-3, -3, -3, -1, -1, -1, 0, ...]
   thrust::inclusive_scan_by_key(scan_row_indices.begin(), 
				 scan_row_indices.end(),
			  	 column_indices.begin(), 
			  	 column_indices.begin(), 
				 thrust::equal_to<IndexType>(),
			  	 thrust::maximum<IndexType>());

   // enumerate the offsets within each column, e.g. [0, 1, 2, 0, 1, 2, 0, ...]
   cusp::array1d<IndexType, cusp::device_memory> column_offsets(num_entries);
   thrust::transform(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(num_entries), 
                     thrust::constant_iterator<IndexType>(pitch), 
                     column_offsets.begin(), 
                     thrust::modulus<IndexType>());

   // sum the indices and the offsets for the final column_indices array 
   cusp::blas::axpy( column_offsets, column_indices, IndexType(1) );

   cusp::array1d<IndexType, cusp::device_memory> row_indices(num_entries);

   // copy valid entries to COO format
   thrust::copy_if
     (thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices.begin(), src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_diagonals))),
      thrust::make_permutation_iterator(thrust::make_zip_iterator(thrust::make_tuple(row_indices_begin, column_indices.begin(), src.values.values.begin())), thrust::make_transform_iterator(thrust::counting_iterator<IndexType>(0), transpose_index_functor<IndexType>(pitch, num_diagonals))) + num_entries,
      thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), dst.column_indices.begin(), dst.values.begin())),
      is_valid_coo_index<IndexType,ValueType>(src.num_rows,src.num_cols));

    row_indices.resize( src.num_entries );
    cusp::detail::indices_to_offsets( row_indices, dst.row_offsets );
}


/////////
// DIA //
/////////
template <typename Matrix1, typename Matrix2>
void coo_to_dia(const Matrix1& src, Matrix2& dst,
                const size_t alignment = 32)
{
    typedef typename Matrix2::index_type IndexType;
    typedef typename Matrix2::value_type ValueType;

    // compute number of occupied diagonals and enumerate them
    cusp::array1d<IndexType,cusp::device_memory> diag_map(src.num_entries);
    thrust::transform(thrust::make_zip_iterator( thrust::make_tuple( src.row_indices.begin(), src.column_indices.begin() ) ), 
		      thrust::make_zip_iterator( thrust::make_tuple( src.row_indices.end()  , src.column_indices.end() ) )  ,
		      diag_map.begin(),
		      occupied_diagonal_functor<IndexType>(src.num_rows)); 

    // place ones in diagonals array locations with occupied diagonals
    cusp::array1d<IndexType,cusp::device_memory> diagonals(src.num_rows+src.num_cols,IndexType(0));
    thrust::scatter(thrust::constant_iterator<IndexType>(1), 
		    thrust::constant_iterator<IndexType>(1)+src.num_entries, 
		    diag_map.begin(),
		    diagonals.begin());

    const IndexType num_diagonals = thrust::reduce(diagonals.begin(), diagonals.end());

    // allocate DIA structure
    dst.resize(src.num_rows, src.num_cols, src.num_entries, num_diagonals, alignment);

    // fill in values array
    thrust::fill(dst.values.values.begin(), dst.values.values.end(), ValueType(0));

    // fill in diagonal_offsets array
    thrust::copy_if(thrust::counting_iterator<IndexType>(0), 
		    thrust::counting_iterator<IndexType>(src.num_rows+src.num_cols),
		    diagonals.begin(),
		    dst.diagonal_offsets.begin(), 
		    is_positive<IndexType>()); 

    // replace shifted diagonals with index of diagonal in offsets array
    cusp::array1d<IndexType,cusp::host_memory> diagonal_offsets( dst.diagonal_offsets );
    for( IndexType num_diag = 0; num_diag < num_diagonals; num_diag++ )
	thrust::replace(diag_map.begin(), diag_map.end(), diagonal_offsets[num_diag], num_diag);

    // copy values to dst
    thrust::scatter(src.values.begin(), src.values.end(),
		    thrust::make_transform_iterator(
				thrust::make_zip_iterator( thrust::make_tuple( src.row_indices.begin(), diag_map.begin() ) ), 
				diagonal_index_functor<IndexType>(dst.values.pitch)), 
                    dst.values.values.begin());

    // shift diagonal_offsets by num_rows 
    cusp::blas::axpy(thrust::constant_iterator<IndexType>(dst.num_rows),
		     thrust::constant_iterator<IndexType>(dst.num_rows)+num_diagonals,
		     dst.diagonal_offsets.begin(),
		     IndexType(-1));
}

template <typename Matrix1, typename Matrix2>
void csr_to_dia(const Matrix1& src, Matrix2& dst,
                const size_t alignment = 32)
{
    typedef typename Matrix2::index_type IndexType;
    typedef typename Matrix2::value_type ValueType;

    // compute number of occupied diagonals and enumerate them
    cusp::array1d<IndexType,cusp::device_memory> row_indices(src.num_entries);
    cusp::detail::offsets_to_indices(src.row_offsets, row_indices);

    cusp::array1d<IndexType,cusp::device_memory> diag_map(src.num_entries);
    thrust::transform(thrust::make_zip_iterator( thrust::make_tuple( row_indices.begin(), src.column_indices.begin() ) ), 
		      thrust::make_zip_iterator( thrust::make_tuple( row_indices.end()  , src.column_indices.end() ) )  ,
		      diag_map.begin(),
		      occupied_diagonal_functor<IndexType>(src.num_rows)); 

    // place ones in diagonals array locations with occupied diagonals
    cusp::array1d<IndexType,cusp::device_memory> diagonals(src.num_rows+src.num_cols,IndexType(0));
    thrust::scatter(thrust::constant_iterator<IndexType>(1), 
		    thrust::constant_iterator<IndexType>(1)+src.num_entries, 
		    diag_map.begin(),
		    diagonals.begin());

    const IndexType num_diagonals = thrust::reduce(diagonals.begin(), diagonals.end());

    // allocate DIA structure
    dst.resize(src.num_rows, src.num_cols, src.num_entries, num_diagonals, alignment);

    // fill in values array
    thrust::fill(dst.values.values.begin(), dst.values.values.end(), ValueType(0));

    // fill in diagonal_offsets array
    thrust::copy_if(thrust::counting_iterator<IndexType>(0), 
		    thrust::counting_iterator<IndexType>(src.num_rows+src.num_cols),
		    diagonals.begin(),
		    dst.diagonal_offsets.begin(), 
		    is_positive<IndexType>()); 

    // replace shifted diagonals with index of diagonal in offsets array
    cusp::array1d<IndexType,cusp::host_memory> diagonal_offsets( dst.diagonal_offsets );
    for( IndexType num_diag = 0; num_diag < num_diagonals; num_diag++ )
	thrust::replace(diag_map.begin(), diag_map.end(), diagonal_offsets[num_diag], num_diag);

    // copy values to dst
    thrust::scatter(src.values.begin(), src.values.end(),
		    thrust::make_transform_iterator(
				thrust::make_zip_iterator( thrust::make_tuple( row_indices.begin(), diag_map.begin() ) ), 
				diagonal_index_functor<IndexType>(dst.values.pitch)), 
                    dst.values.values.begin());

    // shift diagonal_offsets by num_rows 
    cusp::blas::axpy(thrust::constant_iterator<IndexType>(dst.num_rows),
		     thrust::constant_iterator<IndexType>(dst.num_rows)+num_diagonals,
		     dst.diagonal_offsets.begin(),
		     IndexType(-1));
}

/////////
// ELL //
/////////
template <typename Matrix1, typename Matrix2>
void coo_to_ell(const Matrix1& src, Matrix2& dst,
                const size_t num_entries_per_row, const size_t alignment = 32)
{
  typedef typename Matrix2::index_type IndexType;
  typedef typename Matrix2::value_type ValueType;

  if (src.num_entries == 0)
  {
    dst.resize(src.num_rows, src.num_cols, src.num_entries, 0);
    return;
  }

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
void csr_to_ell(const Matrix1& src, Matrix2& dst,
                const size_t num_entries_per_row, const size_t alignment = 32)
{
  typedef typename Matrix2::index_type IndexType;
  typedef typename Matrix2::value_type ValueType;

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


} // end namespace device
} // end namespace detail
} // end namespace cusp

