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

#include <cusp/format.h>

// TODO replace with detail/array2d_utils.h or something
#include <cusp/array2d.h>

#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace cusp
{
namespace detail
{

template <typename T1, typename T2>
void copy_matrix_dimensions(const T1& src, T2& dst)
{
  dst.num_rows    = src.num_rows;
  dst.num_cols    = src.num_cols;
  dst.num_entries = src.num_entries;
}

template <typename T1, typename T2>
void copy(const T1& src, T2& dst,
          cusp::coo_format,
          cusp::coo_format)
{
  copy_matrix_dimensions(src, dst);
  cusp::copy(src.row_indices,    dst.row_indices);
  cusp::copy(src.column_indices, dst.column_indices);
  cusp::copy(src.values,         dst.values);
}

template <typename T1, typename T2>
void copy(const T1& src, T2& dst,
          cusp::csr_format,
          cusp::csr_format)
{    
  copy_matrix_dimensions(src, dst);
  cusp::copy(src.row_offsets,    dst.row_offsets);
  cusp::copy(src.column_indices, dst.column_indices);
  cusp::copy(src.values,         dst.values);
}

template <typename T1, typename T2>
void copy(const T1& src, T2& dst,
          cusp::dia_format,
          cusp::dia_format)
{    
  copy_matrix_dimensions(src, dst);
  cusp::copy(src.diagonal_offsets, dst.diagonal_offsets);
  cusp::copy(src.values,           dst.values);
}

template <typename T1, typename T2>
void copy(const T1& src, T2& dst,
          cusp::ell_format,
          cusp::ell_format)
{    
  copy_matrix_dimensions(src, dst);
  cusp::copy(src.column_indices, dst.column_indices);
  cusp::copy(src.values,         dst.values);
}

template <typename T1, typename T2>
void copy(const T1& src, T2& dst,
          cusp::hyb_format,
          cusp::hyb_format)
{    
  copy_matrix_dimensions(src, dst);
  cusp::copy(src.ell, dst.ell);
  cusp::copy(src.coo, dst.coo);
}

template <typename T1, typename T2>
void copy(const T1& src, T2& dst,
          cusp::array1d_format,
          cusp::array1d_format)
{    
  dst.resize(src.size());
  thrust::copy(src.begin(), src.end(), dst.begin());
}

  
// same orientation
template <typename T1, typename T2, typename Orientation>
void copy_array2d(const T1& src, T2& dst,
                  Orientation)
{
  copy_matrix_dimensions(src, dst);
  cusp::copy(src.values, dst.values);
}


// convert a linear index to a linear index in the transpose
template <typename T, typename SourceOrientation, typename DestinationOrientation>
struct gather_index : public thrust::unary_function<T, T>
{
    T m, n; // destination dimensions

    __host__ __device__
    gather_index(T _m, T _n) : m(_m), n(_n) {}

    __host__ __device__
    T operator()(T linear_index)
    {
        T i = cusp::detail::linear_index_to_row_index(linear_index, m, n, DestinationOrientation());
        T j = cusp::detail::linear_index_to_col_index(linear_index, m, n, DestinationOrientation());

        // TODO use real pitch
        T pitch = cusp::detail::minor_dimension(m, n, SourceOrientation());
        
        return cusp::detail::index_of(i, j, pitch, SourceOrientation());
    }
};

template <typename T1, typename T2, typename SourceOrientation, typename DestinationOrientation>
void copy_array2d(const T1& src, T2& dst,
                  SourceOrientation,
                  DestinationOrientation)
{
  copy_matrix_dimensions(src, dst);
  dst.resize(src.num_rows, src.num_cols);
   
  // TODO support non-trivial stride
  thrust::counting_iterator<size_t> begin(0);
  thrust::counting_iterator<size_t> end(src.values.size());
  
#if THRUST_VERSION >= 100300
  thrust::gather(thrust::make_transform_iterator(begin, gather_index<size_t, SourceOrientation, DestinationOrientation>(dst.num_rows, dst.num_cols)),
                 thrust::make_transform_iterator(end,   gather_index<size_t, SourceOrientation, DestinationOrientation>(dst.num_rows, dst.num_cols)),
                 src.values.begin(),
                 dst.values.begin());
#else
  // TODO remove this when Thrust v1.2.x is unsupported
  thrust::next::gather(thrust::make_transform_iterator(begin, gather_index<size_t, SourceOrientation, DestinationOrientation>(dst.num_rows, dst.num_cols)),
                       thrust::make_transform_iterator(end,   gather_index<size_t, SourceOrientation, DestinationOrientation>(dst.num_rows, dst.num_cols)),
                       src.values.begin(),
                       dst.values.begin());
#endif
}

template <typename T1, typename T2>
void copy(const T1& src, T2& dst,
          cusp::array2d_format,
          cusp::array2d_format)
{
  copy_array2d(src, dst,
      typename T1::orientation(),
      typename T2::orientation());
}

} // end namespace detail


/////////////////
// Entry Point //
/////////////////

template <typename T1, typename T2>
void copy(const T1& src, T2& dst)
{
  cusp::detail::copy(src, dst, typename T1::format(), typename T2::format());
}

} // end namespace cusp

