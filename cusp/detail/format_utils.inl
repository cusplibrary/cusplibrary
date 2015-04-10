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

#include <cusp/array1d.h>
#include <cusp/array2d.h>

#include <cusp/detail/format.h>
#include <cusp/detail/functional.h>

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/inner_product.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

namespace cusp
{
namespace detail
{

template <typename DerivedPolicy, typename Matrix, typename Array>
void extract_diagonal(thrust::execution_policy<DerivedPolicy> &exec,
                      const Matrix& A,
                      Array& output,
                      cusp::coo_format)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Array::value_type   ValueType;

    // initialize output to zero
    thrust::fill(exec, output.begin(), output.end(), ValueType(0));

    // scatter the diagonal values to output
    thrust::scatter_if(exec,
                       A.values.begin(), A.values.end(),
                       A.row_indices.begin(),
                       thrust::make_transform_iterator(
                           thrust::make_zip_iterator(
                               thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin())),
                           cusp::detail::equal_tuple_functor<IndexType>()),
                       output.begin());
}

template <typename DerivedPolicy, typename Matrix, typename Array>
void extract_diagonal(thrust::execution_policy<DerivedPolicy> &exec,
                      const Matrix& A,
                      Array& output,
                      cusp::csr_format)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Array::value_type   ValueType;
    typedef typename Array::memory_space MemorySpace;

    // first expand the compressed row offsets into row indices
    thrust::detail::temporary_array<IndexType, DerivedPolicy> row_indices(exec, A.num_entries);
    cusp::offsets_to_indices(exec, A.row_offsets, row_indices);

    // initialize output to zero
    thrust::fill(exec, output.begin(), output.end(), ValueType(0));

    // scatter the diagonal values to output
    thrust::scatter_if(exec,
                       A.values.begin(), A.values.end(),
                       row_indices.begin(),
                       thrust::make_transform_iterator(
                           thrust::make_zip_iterator(
                               thrust::make_tuple(row_indices.begin(), A.column_indices.begin())),
                           cusp::detail::equal_tuple_functor<IndexType>()),
                       output.begin());
}

template <typename DerivedPolicy, typename Matrix, typename Array>
void extract_diagonal(thrust::execution_policy<DerivedPolicy> &exec,
                      const Matrix& A,
                      Array& output,
                      cusp::dia_format)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Array::value_type   ValueType;

    for(size_t i = 0; i < A.diagonal_offsets.size(); i++)
    {
        if(A.diagonal_offsets[i] == 0)
        {
            // diagonal found, copy to output and return
            thrust::copy(exec,
                         A.values.values.begin() + A.values.pitch * i,
                         A.values.values.begin() + A.values.pitch * i + output.size(),
                         output.begin());
            return;
        }
    }

    // no diagonal found
    thrust::fill(exec, output.begin(), output.end(), ValueType(0));
}

template <typename DerivedPolicy, typename Matrix, typename Array>
void extract_diagonal(thrust::execution_policy<DerivedPolicy> &exec,
                      const Matrix& A,
                      Array& output,
                      cusp::ell_format)
{
    typedef typename Matrix::index_type  IndexType;
    typedef typename Array::value_type   ValueType;

    // initialize output to zero
    thrust::fill(exec, output.begin(), output.end(), ValueType(0));

    thrust::scatter_if
    (exec,
     A.values.values.begin(), A.values.values.end(),
     thrust::make_transform_iterator(thrust::counting_iterator<size_t>(0),
                                     cusp::detail::modulus_value<size_t>(A.column_indices.pitch)),
     thrust::make_zip_iterator(thrust::make_tuple
                               (thrust::make_transform_iterator(
                                    thrust::counting_iterator<size_t>(0),
                                    cusp::detail::modulus_value<size_t>(A.column_indices.pitch)),
                                A.column_indices.values.begin())),
     output.begin(),
     cusp::detail::equal_tuple_functor<IndexType>());
}

template <typename DerivedPolicy, typename Matrix, typename Array>
void extract_diagonal(thrust::execution_policy<DerivedPolicy> &exec,
                      const Matrix& A,
                      Array& output,
                      cusp::hyb_format)
{
    typedef typename Matrix::index_type  IndexType;

    // extract COO diagonal
    cusp::extract_diagonal(exec, A.coo, output);

    // extract ELL diagonal
    thrust::scatter_if
    (exec,
     A.ell.values.values.begin(), A.ell.values.values.end(),
     thrust::make_transform_iterator(
         thrust::counting_iterator<size_t>(0), cusp::detail::modulus_value<size_t>(A.ell.column_indices.pitch)),
     thrust::make_zip_iterator(thrust::make_tuple(
                                   thrust::make_transform_iterator(
                                       thrust::counting_iterator<size_t>(0), cusp::detail::modulus_value<size_t>(A.ell.column_indices.pitch)),
                                   A.ell.column_indices.values.begin())),
     output.begin(),
     cusp::detail::equal_tuple_functor<IndexType>());
}

template <typename DerivedPolicy, typename OffsetArray, typename IndexArray>
void offsets_to_indices(thrust::execution_policy<DerivedPolicy> &exec,
                        const OffsetArray& offsets, IndexArray& indices)
{
    typedef typename OffsetArray::value_type OffsetType;

    // convert compressed row offsets into uncompressed row indices
    thrust::fill(exec, indices.begin(), indices.end(), OffsetType(0));
    thrust::scatter_if( exec,
                        thrust::counting_iterator<OffsetType>(0),
                        thrust::counting_iterator<OffsetType>(offsets.size()-1),
                        offsets.begin(),
                        thrust::make_transform_iterator(
                            thrust::make_zip_iterator( thrust::make_tuple( offsets.begin(), offsets.begin()+1 ) ),
                            cusp::detail::not_equal_tuple_functor<OffsetType>()),
                        indices.begin());
    thrust::inclusive_scan(exec, indices.begin(), indices.end(), indices.begin(), thrust::maximum<OffsetType>());
}

template <typename DerivedPolicy, typename IndexArray, typename OffsetArray>
void indices_to_offsets(thrust::execution_policy<DerivedPolicy> &exec,
                        const IndexArray& indices, OffsetArray& offsets)
{
    typedef typename OffsetArray::value_type OffsetType;

    // convert uncompressed row indices into compressed row offsets
    thrust::lower_bound(exec,
                        indices.begin(),
                        indices.end(),
                        thrust::counting_iterator<OffsetType>(0),
                        thrust::counting_iterator<OffsetType>(offsets.size()),
                        offsets.begin());
}

template <typename DerivedPolicy, typename ArrayType1, typename ArrayType2>
size_t count_diagonals(thrust::execution_policy<DerivedPolicy> &exec,
                       const size_t num_rows,
                       const size_t num_cols,
                       const ArrayType1& row_indices,
                       const ArrayType2& column_indices )
{
    typedef typename ArrayType1::value_type IndexType;
    typedef typename ArrayType1::memory_space MemorySpace;

    size_t num_entries = row_indices.size();

    thrust::detail::temporary_array<IndexType, DerivedPolicy> values(exec, num_rows + num_cols);
    thrust::fill(exec, values.begin(), values.end(), IndexType(0));

    thrust::scatter(exec,
                    thrust::constant_iterator<IndexType>(1),
                    thrust::constant_iterator<IndexType>(1)+num_entries,
                    thrust::make_transform_iterator(
                        thrust::make_zip_iterator(
                            thrust::make_tuple( row_indices.begin(), column_indices.begin() ) ),
                        cusp::detail::occupied_diagonal_functor<IndexType>(num_rows)),
                    values.begin());

    return thrust::reduce(exec, values.begin(), values.end());
}


template <typename DerivedPolicy, typename ArrayType>
size_t compute_max_entries_per_row(thrust::execution_policy<DerivedPolicy> &exec,
                                   const ArrayType& row_offsets)
{
    typedef typename ArrayType::value_type IndexType;

    size_t max_entries_per_row =
        thrust::inner_product(exec,
                              row_offsets.begin() + 1, row_offsets.end(),
                              row_offsets.begin(),
                              IndexType(0),
                              thrust::maximum<IndexType>(),
                              thrust::minus<IndexType>());

    return max_entries_per_row;
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
size_t compute_optimal_entries_per_row(thrust::execution_policy<DerivedPolicy> &exec,
                                       const ArrayType& row_offsets,
                                       float relative_speed,
                                       size_t breakeven_threshold)
{
    typedef typename ArrayType::value_type IndexType;
    typedef typename ArrayType::memory_space MemorySpace;

    const size_t num_rows = row_offsets.size()-1;

    // compute maximum row length
    IndexType max_cols_per_row = compute_max_entries_per_row(exec, row_offsets);

    // allocate storage for the cumulative histogram and histogram
    thrust::detail::temporary_array<IndexType, DerivedPolicy> cumulative_histogram(exec, max_cols_per_row + 1);
    thrust::fill(exec, cumulative_histogram.begin(), cumulative_histogram.end(), IndexType(0));

    // compute distribution of nnz per row
    thrust::detail::temporary_array<IndexType, DerivedPolicy> entries_per_row(exec, num_rows);
    thrust::adjacent_difference(exec, row_offsets.begin()+1, row_offsets.end(), entries_per_row.begin());

    // sort data to bring equal elements together
    thrust::sort(exec, entries_per_row.begin(), entries_per_row.end());

    // find the end of each bin of values
    thrust::counting_iterator<IndexType> search_begin(0);
    thrust::upper_bound(exec,
                        entries_per_row.begin(),
                        entries_per_row.end(),
                        search_begin,
                        search_begin + max_cols_per_row + 1,
                        cumulative_histogram.begin());

    // compute optimal ELL column size
    IndexType num_cols_per_row = thrust::find_if(exec,
                                 cumulative_histogram.begin(), cumulative_histogram.end()-1,
                                 cusp::detail::speed_threshold_functor(num_rows, relative_speed, breakeven_threshold))
                                 - cumulative_histogram.begin();

    return num_cols_per_row;
}

} // end detail namespace

template <typename DerivedPolicy, typename Matrix, typename Array>
void extract_diagonal(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                      const Matrix& A, Array& output)
{
    typedef typename Matrix::format Format;

    output.resize(thrust::min(A.num_rows, A.num_cols));

    // dispatch on matrix format
    detail::extract_diagonal(thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
                             A, output, Format());
}

template <typename Matrix, typename Array>
void extract_diagonal(const Matrix& A, Array& output)
{
    using thrust::system::detail::generic::select_system;

    typedef typename Matrix::memory_space System1;
    typedef typename Array::memory_space  System2;

    System1 system1;
    System2 system2;

    extract_diagonal(select_system(system1,system2), A, output);
}

template <typename DerivedPolicy, typename OffsetArray, typename IndexArray>
void offsets_to_indices(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                        const OffsetArray& offsets, IndexArray& indices)
{
    return detail::offsets_to_indices(thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
                                      offsets, indices);
}

template <typename OffsetArray, typename IndexArray>
void offsets_to_indices(const OffsetArray& offsets, IndexArray& indices)
{
    using thrust::system::detail::generic::select_system;

    typedef typename IndexArray::memory_space  System1;
    typedef typename OffsetArray::memory_space System2;

    System1 system1;
    System2 system2;

    return offsets_to_indices(select_system(system1,system2), offsets, indices);
}

template <typename DerivedPolicy, typename IndexArray, typename OffsetArray>
void indices_to_offsets(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                        const IndexArray& indices, OffsetArray& offsets)
{
    return detail::indices_to_offsets(thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
                                      indices, offsets);
}

template <typename IndexArray, typename OffsetArray>
void indices_to_offsets(const IndexArray& indices, OffsetArray& offsets)
{
    using thrust::system::detail::generic::select_system;

    typedef typename IndexArray::memory_space  System1;
    typedef typename OffsetArray::memory_space System2;

    System1 system1;
    System2 system2;

    return indices_to_offsets(select_system(system1,system2), indices, offsets);
}

template <typename DerivedPolicy, typename ArrayType1, typename ArrayType2>
size_t count_diagonals(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                       const size_t num_rows,
                       const size_t num_cols,
                       const ArrayType1& row_indices,
                       const ArrayType2& column_indices)
{
    return detail::count_diagonals(thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
                                   num_rows, num_cols, row_indices, column_indices);
}

template <typename ArrayType1, typename ArrayType2>
size_t count_diagonals(const size_t num_rows,
                       const size_t num_cols,
                       const ArrayType1& row_indices,
                       const ArrayType2& column_indices)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType1::memory_space System1;
    typedef typename ArrayType2::memory_space System2;

    System1 system1;
    System2 system2;

    return count_diagonals(select_system(system1,system2), num_rows, num_cols, row_indices, column_indices);
}

template <typename DerivedPolicy, typename ArrayType>
size_t compute_max_entries_per_row(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                   const ArrayType& row_offsets)
{
    return detail::compute_max_entries_per_row(thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
            row_offsets);
}

template <typename ArrayType>
size_t compute_max_entries_per_row(const ArrayType& row_offsets)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType::memory_space System;

    System system;

    return compute_max_entries_per_row(select_system(system), row_offsets);
}

template <typename DerivedPolicy, typename ArrayType>
size_t compute_optimal_entries_per_row(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                       const ArrayType& row_offsets,
                                       float relative_speed,
                                       size_t breakeven_threshold)
{
    return detail::compute_optimal_entries_per_row(thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
            row_offsets, relative_speed, breakeven_threshold);
}

template <typename ArrayType>
size_t compute_optimal_entries_per_row(const ArrayType& row_offsets,
                                       float relative_speed,
                                       size_t breakeven_threshold)
{
    using thrust::system::detail::generic::select_system;

    typedef typename ArrayType::memory_space System;

    System system;

    return compute_optimal_entries_per_row(select_system(system), row_offsets, relative_speed, breakeven_threshold);
}

} // end namespace cusp

