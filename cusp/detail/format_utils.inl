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
namespace detail
{

template <typename IndexType>
struct empty_row_functor
{
    typedef bool result_type;

    template <typename Tuple>
    __host__ __device__
    bool operator()(const Tuple& t) const
    {
        const IndexType a = thrust::get<0>(t);
        const IndexType b = thrust::get<1>(t);

        return a != b;
    }
};

template<typename IndexType>
struct row_operator : public std::unary_function<size_t,IndexType>
{
    size_t pitch;

    row_operator(size_t pitch)
        : pitch(pitch) {}

    __host__ __device__
    IndexType operator()(const size_t & linear_index) const
    {
        return linear_index % pitch;
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

        return j-i+num_rows;
    }
};

struct speed_threshold_functor
{
    size_t num_rows;
    float  relative_speed;
    size_t breakeven_threshold;

    speed_threshold_functor(const size_t num_rows, const float relative_speed, const size_t breakeven_threshold)
        : num_rows(num_rows),
          relative_speed(relative_speed),
          breakeven_threshold(breakeven_threshold)
    {}

    template <typename IndexType>
    __host__ __device__
    bool operator()(const IndexType rows) const
    {
        return relative_speed * (num_rows-rows) < num_rows || (size_t) (num_rows-rows) < breakeven_threshold;
    }
};


template <typename IndexType>
struct tuple_equal_to : public thrust::unary_function<thrust::tuple<IndexType,IndexType>,bool>
{
    __host__ __device__
    bool operator()(const thrust::tuple<IndexType,IndexType>& t) const
    {
        return thrust::get<0>(t) == thrust::get<1>(t);
    }
};

template <typename OffsetArray, typename IndexArray>
void offsets_to_indices(const OffsetArray& offsets, IndexArray& indices)
{
    typedef typename OffsetArray::value_type OffsetType;

    // convert compressed row offsets into uncompressed row indices
    thrust::fill(indices.begin(), indices.end(), OffsetType(0));
    thrust::scatter_if( thrust::counting_iterator<OffsetType>(0),
                        thrust::counting_iterator<OffsetType>(offsets.size()-1),
                        offsets.begin(),
                        thrust::make_transform_iterator(
                            thrust::make_zip_iterator( thrust::make_tuple( offsets.begin(), offsets.begin()+1 ) ),
                            empty_row_functor<OffsetType>()),
                        indices.begin());
    thrust::inclusive_scan(indices.begin(), indices.end(), indices.begin(), thrust::maximum<OffsetType>());
}

template <typename IndexArray, typename OffsetArray>
void indices_to_offsets(const IndexArray& indices, OffsetArray& offsets)
{
    typedef typename OffsetArray::value_type OffsetType;

    // convert uncompressed row indices into compressed row offsets
    thrust::lower_bound(indices.begin(),
                        indices.end(),
                        thrust::counting_iterator<OffsetType>(0),
                        thrust::counting_iterator<OffsetType>(offsets.size()),
                        offsets.begin());
}

template <typename Matrix, typename Array>
void extract_diagonal(const Matrix& A, Array& output, cusp::coo_format)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Array::value_type   ValueType;

    // initialize output to zero
    thrust::fill(output.begin(), output.end(), ValueType(0));

    // scatter the diagonal values to output
    thrust::scatter_if(A.values.begin(), A.values.end(),
                       A.row_indices.begin(),
                       thrust::make_transform_iterator(
                           thrust::make_zip_iterator(
                               thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin())),
                           tuple_equal_to<IndexType>()),
                       output.begin());
}

template <typename Matrix, typename Array>
void extract_diagonal(const Matrix& A, Array& output, cusp::csr_format)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Array::value_type   ValueType;
    typedef typename Array::memory_space MemorySpace;

    // first expand the compressed row offsets into row indices
    cusp::array1d<IndexType,MemorySpace> row_indices(A.num_entries);
    offsets_to_indices(A.row_offsets, row_indices);

    // initialize output to zero
    thrust::fill(output.begin(), output.end(), ValueType(0));

    // scatter the diagonal values to output
    thrust::scatter_if(A.values.begin(), A.values.end(),
                       row_indices.begin(),
                       thrust::make_transform_iterator(
                           thrust::make_zip_iterator(
                               thrust::make_tuple(row_indices.begin(), A.column_indices.begin())),
                           tuple_equal_to<IndexType>()),
                       output.begin());
}

template <typename Matrix, typename Array>
void extract_diagonal(const Matrix& A, Array& output, cusp::dia_format)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Array::value_type   ValueType;

    // copy diagonal_offsets to host (sometimes unnecessary)
    cusp::array1d<IndexType,cusp::host_memory> diagonal_offsets(A.diagonal_offsets);

    for(size_t i = 0; i < diagonal_offsets.size(); i++)
    {
        if(diagonal_offsets[i] == 0)
        {
            // diagonal found, copy to output and return
            thrust::copy(A.values.values.begin() + A.values.pitch * i,
                         A.values.values.begin() + A.values.pitch * i + output.size(),
                         output.begin());
            return;
        }
    }

    // no diagonal found
    thrust::fill(output.begin(), output.end(), ValueType(0));
}

template <typename Matrix, typename Array>
void extract_diagonal(const Matrix& A, Array& output, cusp::ell_format)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Array::value_type   ValueType;

    // initialize output to zero
    thrust::fill(output.begin(), output.end(), ValueType(0));

    thrust::scatter_if
    (A.values.values.begin(), A.values.values.end(),
     thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), row_operator<IndexType>(A.column_indices.pitch)),
     thrust::make_zip_iterator(thrust::make_tuple
                               (thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), row_operator<IndexType>(A.column_indices.pitch)),
                                A.column_indices.values.begin())),
     output.begin(),
     tuple_equal_to<IndexType>());

    // TODO ignore padded values in column_indices
}

template <typename Matrix, typename Array>
void extract_diagonal(const Matrix& A, Array& output, cusp::hyb_format)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Array::value_type   ValueType;

    // extract COO diagonal
    cusp::detail::extract_diagonal(A.coo, output);

    // extract ELL diagonal
    thrust::scatter_if
    (A.ell.values.values.begin(), A.ell.values.values.end(),
     thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), row_operator<IndexType>(A.ell.column_indices.pitch)),
     thrust::make_zip_iterator(thrust::make_tuple
                               (thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0), row_operator<IndexType>(A.ell.column_indices.pitch)),
                                A.ell.column_indices.values.begin())),
     output.begin(),
     tuple_equal_to<IndexType>());

    // TODO ignore padded values in column_indices
}

template <typename Matrix, typename Array>
void extract_diagonal(const Matrix& A, Array& output)
{
    output.resize(thrust::min(A.num_rows, A.num_cols));

    // dispatch on matrix format
    extract_diagonal(A, output, typename Matrix::format());
}

template <typename DerivedPolicy, typename ArrayType1, typename ArrayType2>
size_t count_diagonals(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                       const size_t num_rows,
                       const size_t num_cols,
                       const size_t num_entries,
                       const ArrayType1& row_indices,
                       const ArrayType2& column_indices )
{
    typedef typename ArrayType1::value_type IndexType;

    cusp::array1d<IndexType,cusp::device_memory> values(num_rows+num_cols,IndexType(0));

    thrust::scatter(exec,
                    thrust::constant_iterator<IndexType>(1),
                    thrust::constant_iterator<IndexType>(1)+num_entries,
                    thrust::make_transform_iterator(thrust::make_zip_iterator( thrust::make_tuple( row_indices.begin(), column_indices.begin() ) ),
                            occupied_diagonal_functor<IndexType>(num_rows)),
                    values.begin());

    return thrust::reduce(exec, values.begin(), values.end());
}

template <typename ArrayType1, typename ArrayType2>
size_t count_diagonals(const size_t num_rows,
                       const size_t num_cols,
                       const size_t num_entries,
                       const ArrayType1& row_indices,
                       const ArrayType2& column_indices )
{
  using thrust::system::detail::generic::select_system;

  typedef typename ArrayType::memory_space System;

  System system;

  return count_diagonals(select_system(system), num_rows, num_cols, num_entries, row_indices, column_indices);
}

template <typename DerivedPolicy, typename ArrayType>
size_t compute_max_entries_per_row(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                   const ArrayType& row_offsets)
{
    typedef typename ArrayType::value_type IndexType;

    size_t max_entries_per_row =
        thrust::inner_product(exec, row_offsets.begin() + 1, row_offsets.end(),
                              row_offsets.begin(),
                              IndexType(0),
                              thrust::maximum<IndexType>(),
                              thrust::minus<IndexType>());

    return max_entries_per_row;
}

template <typename DerivedPolicy, typename ArrayType>
size_t compute_max_entries_per_row(const ArrayType& row_offsets )
{
  using thrust::system::detail::generic::select_system;

  typedef typename ArrayType::memory_space System;

  System system;

  return compute_max_entries_per_row(select_system(system), row_offsets);
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
template <typename DerivedPolicy, typename ArrayType>
size_t compute_optimal_entries_per_row(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                       const ArrayType& row_offsets,
                                       float relative_speed = 3.0f,
                                       size_t breakeven_threshold = 4096)
{
    typedef typename ArrayType::value_type IndexType;

    const size_t num_rows = row_offsets.size()-1;

    // compute maximum row length
    IndexType max_cols_per_row = compute_max_entries_per_row(row_offsets);

    // allocate storage for the cumulative histogram and histogram
    cusp::array1d<IndexType,cusp::device_memory> cumulative_histogram(max_cols_per_row + 1, IndexType(0));

    // compute distribution of nnz per row
    cusp::array1d<IndexType,cusp::device_memory> entries_per_row(num_rows);
    thrust::adjacent_difference( row_offsets.begin()+1, row_offsets.end(), entries_per_row.begin() );

    // sort data to bring equal elements together
    thrust::sort(entries_per_row.begin(), entries_per_row.end());

    // find the end of each bin of values
    thrust::counting_iterator<IndexType> search_begin(0);
    thrust::upper_bound(entries_per_row.begin(),
                        entries_per_row.end(),
                        search_begin,
                        search_begin + max_cols_per_row + 1,
                        cumulative_histogram.begin());

    // compute optimal ELL column size
    IndexType num_cols_per_row = thrust::find_if( cumulative_histogram.begin(), cumulative_histogram.end()-1,
                                 speed_threshold_functor(num_rows, relative_speed, breakeven_threshold) )
                                 - cumulative_histogram.begin();

    return num_cols_per_row;
}

template <typename ArrayType>
size_t compute_optimal_entries_per_row(const ArrayType& row_offsets,
                                       float relative_speed = 3.0f,
                                       size_t breakeven_threshold = 4096)
{
  using thrust::system::detail::generic::select_system;

  typedef typename ArrayType::memory_space System;

  System system;

  return compute_optimal_entries_per_row(select_system(system), row_offsets, relative_speed, breakeven_threshold);
}


} // end namespace detail
} // end namespace cusp
