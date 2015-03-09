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

#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>
#include <cusp/detail/format.h>
#include <cusp/sort.h>
#include <cusp/verify.h>

#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <thrust/iterator/zip_iterator.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy,
         typename MatrixType1, typename MatrixType2, typename MatrixType3,
         typename BinaryFunction>
void elementwise(thrust::execution_policy<DerivedPolicy>& exec,
                 const MatrixType1& A, const MatrixType2& B, MatrixType3& C,
                 BinaryFunction op,
                 cusp::array2d_format& format)
{
    C.resize(A.num_rows, A.num_cols);

    thrust::transform(A.values.begin(), A.values.end(),
                      B.values.begin(),
                      C.values.begin(),
                      op);
}

template <typename DerivedPolicy,
         typename MatrixType1, typename MatrixType2, typename MatrixType3,
         typename BinaryFunction>
void elementwise(thrust::execution_policy<DerivedPolicy>& exec,
                 const MatrixType1& A, const MatrixType2& B, MatrixType3& C,
                 BinaryFunction op,
                 cusp::coo_format&)
{
    using namespace thrust::placeholders;

    typedef typename MatrixType3::index_type   IndexType;
    typedef typename MatrixType3::value_type   ValueType;
    typedef typename MatrixType3::memory_space MemorySpace;

    IndexType A_nnz = A.num_entries;
    IndexType B_nnz = B.num_entries;

    if (A_nnz == 0 && B_nnz == 0)
    {
        C.resize(A.num_rows, A.num_cols, 0);
        return;
    }

    cusp::array1d<IndexType,MemorySpace> rows(A_nnz + B_nnz);
    cusp::array1d<IndexType,MemorySpace> cols(A_nnz + B_nnz);
    cusp::array1d<ValueType,MemorySpace> vals(A_nnz + B_nnz);

    thrust::copy(exec, A.row_indices.begin(),    A.row_indices.end(),    rows.begin());
    thrust::copy(exec, B.row_indices.begin(),    B.row_indices.end(),    rows.begin() + A_nnz);
    thrust::copy(exec, A.column_indices.begin(), A.column_indices.end(), cols.begin());
    thrust::copy(exec, B.column_indices.begin(), B.column_indices.end(), cols.begin() + A_nnz);
    thrust::copy(exec, A.values.begin(),         A.values.end(),         vals.begin());

    // apply transformation to B's values
    if(thrust::detail::is_same< BinaryFunction, thrust::plus<ValueType> >::value)
        thrust::transform(exec, B.values.begin(), B.values.end(), vals.begin() + A_nnz, thrust::identity<ValueType>());
    else
        thrust::transform(exec, B.values.begin(), B.values.end(), vals.begin() + A_nnz, thrust::negate<ValueType>());

    // sort by (I,J)
    cusp::sort_by_row_and_column(exec, rows, cols, vals, 0, A.num_rows, 0, A.num_cols);

    // compute unique number of nonzeros in the output
    IndexType C_nnz = thrust::inner_product(exec,
                                            thrust::make_zip_iterator(thrust::make_tuple(rows.begin(), cols.begin())),
                                            thrust::make_zip_iterator(thrust::make_tuple(rows.end (),  cols.end()))   - 1,
                                            thrust::make_zip_iterator(thrust::make_tuple(rows.begin(), cols.begin())) + 1,
                                            IndexType(1),
                                            thrust::plus<IndexType>(),
                                            thrust::not_equal_to< thrust::tuple<IndexType,IndexType> >());

    // allocate space for output
    C.resize(A.num_rows, A.num_cols, C_nnz);

    // sum values with the same (i,j)
    thrust::reduce_by_key(exec,
                          thrust::make_zip_iterator(thrust::make_tuple(rows.begin(), cols.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(rows.end(),   cols.end())),
                          vals.begin(),
                          thrust::make_zip_iterator(thrust::make_tuple(C.row_indices.begin(), C.column_indices.begin())),
                          C.values.begin(),
                          thrust::equal_to< thrust::tuple<IndexType,IndexType> >(),
                          thrust::plus<ValueType>());

    int num_zeros = thrust::count(exec, C.values.begin(), C.values.end(), ValueType(0));

    // The result of the elementwise operation contains zero entries so we need
    // to contract the result to produce a strictly valid COO matrix
    if(num_zeros != 0)
    {
        int num_reduced_entries =
            thrust::remove_if(
                exec,
                thrust::make_zip_iterator(
                  thrust::make_tuple(C.row_indices.begin(), C.column_indices.begin(), C.values.begin())),
                thrust::make_zip_iterator(
                  thrust::make_tuple(C.row_indices.end(),   C.column_indices.end(), C.values.end())),
                C.values.begin(),
                _1 == ValueType(0)) -
            thrust::make_zip_iterator(
                thrust::make_tuple(C.row_indices.begin(), C.column_indices.begin(), C.values.begin()));

        C.resize(C.num_rows, C.num_cols, num_reduced_entries);
    }
}

template <typename DerivedPolicy,
         typename MatrixType1, typename MatrixType2, typename MatrixType3,
         typename BinaryFunction>
void elementwise(thrust::execution_policy<DerivedPolicy>& exec,
                 const MatrixType1& A, const MatrixType2& B, MatrixType3& C,
                 BinaryFunction op,
                 cusp::csr_format&)
{
    typedef typename MatrixType1::memory_space MemorySpace;

    typedef coo_matrix_view<typename MatrixType1::row_offsets_array_type::view,
            typename MatrixType1::column_indices_array_type::const_view,
            typename MatrixType1::values_array_type::const_view> View1;
    typedef coo_matrix_view<typename MatrixType2::row_offsets_array_type::view,
            typename MatrixType2::column_indices_array_type::const_view,
            typename MatrixType2::values_array_type::const_view> View2;

    typename MatrixType1::row_offsets_array_type::container A_row_indices(A.num_entries);
    typename MatrixType2::row_offsets_array_type::container B_row_indices(B.num_entries);

    cusp::offsets_to_indices(exec, A.row_offsets, A_row_indices);
    cusp::offsets_to_indices(exec, B.row_offsets, B_row_indices);

    View1 A_coo_view(A.num_rows, A.num_cols, A.num_entries,
                     cusp::make_array1d_view(A_row_indices),
                     cusp::make_array1d_view(A.column_indices),
                     cusp::make_array1d_view(A.values));
    View2 B_coo_view(B.num_rows, B.num_cols, B.num_entries,
                     cusp::make_array1d_view(B_row_indices),
                     cusp::make_array1d_view(B.column_indices),
                     cusp::make_array1d_view(B.values));
    typename cusp::detail::as_coo_type<MatrixType3>::type C_coo;

    cusp::elementwise(exec, A_coo_view, B_coo_view, C_coo, op);

    cusp::convert(exec, C_coo, C);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp
