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
namespace elementwise_detail
{

template <typename BinaryFunction>
struct ops
{
    typedef typename BinaryFunction::result_type ValueType;
    typedef thrust::minus<ValueType> Sub;

    typedef typename thrust::detail::eval_if<
    thrust::detail::is_same<Sub, BinaryFunction>::value
    , thrust::detail::identity_< thrust::negate<ValueType> >
    , thrust::detail::identity_< thrust::identity<ValueType> >
    >::type unary_op_type;

    typedef typename thrust::detail::eval_if<
    thrust::detail::is_same<Sub, BinaryFunction>::value
    , thrust::detail::identity_< thrust::plus<ValueType> >
    , thrust::detail::identity_< BinaryFunction >
    >::type binary_op_type;
};

} // end elementwise_detail namespace

template <typename DerivedPolicy,
          typename MatrixType1, typename MatrixType2, typename MatrixType3,
          typename BinaryFunction>
void elementwise(thrust::execution_policy<DerivedPolicy>& exec,
                 const MatrixType1& A, const MatrixType2& B, MatrixType3& C,
                 BinaryFunction op,
                 cusp::array2d_format)
{
    C.resize(A.num_rows, A.num_cols);

    thrust::transform(A.values.begin(),
                      A.values.end(),
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
                 cusp::coo_format)
{
    using namespace thrust::placeholders;

    typedef typename MatrixType3::index_type   IndexType;
    typedef typename MatrixType3::value_type   ValueType;
    typedef typename MatrixType3::memory_space MemorySpace;

    typedef typename MatrixType1::const_coo_view_type CooView1;
    typedef typename MatrixType2::const_coo_view_type CooView2;

    typedef typename CooView1::row_indices_array_type::const_iterator               RowIterator1;
    typedef typename CooView1::column_indices_array_type::const_iterator            ColumnIterator1;
    typedef typename CooView1::values_array_type::const_iterator                    ValueIterator1;
    typedef thrust::tuple<RowIterator1,ColumnIterator1>                             IteratorTuple1;
    typedef thrust::zip_iterator<IteratorTuple1>                                    ZipIterator1;

    typedef typename CooView2::row_indices_array_type::const_iterator               RowIterator2;
    typedef typename CooView2::column_indices_array_type::const_iterator            ColumnIterator2;
    typedef typename CooView2::values_array_type::const_iterator                    ValueIterator2;
    typedef thrust::tuple<RowIterator2,ColumnIterator2>                             IteratorTuple2;
    typedef thrust::zip_iterator<IteratorTuple2>                                    ZipIterator2;

    typedef typename cusp::array1d<IndexType,MemorySpace>::iterator                 IndexIterator;
    typedef cusp::join_iterator<ZipIterator1, ZipIterator2, IndexIterator>          JoinIndexIterator;

    typedef typename elementwise_detail::ops<BinaryFunction>::unary_op_type         UnaryOp;
    typedef typename elementwise_detail::ops<BinaryFunction>::binary_op_type        BinaryOp;
    typedef thrust::transform_iterator<UnaryOp, ValueIterator2>                     TransValueIterator;
    typedef cusp::join_iterator<ValueIterator1, TransValueIterator, IndexIterator>  JoinValueIterator;

    size_t A_nnz = A.num_entries;
    size_t B_nnz = B.num_entries;
    size_t num_entries = A_nnz + B_nnz;

    if (A_nnz == 0 && B_nnz == 0)
    {
        C.resize(A.num_rows, A.num_cols, 0);
        return;
    }

#if THRUST_VERSION >= 100900
    ZipIterator1 A_tuples(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin()));
    ZipIterator2 B_tuples(thrust::make_tuple(B.row_indices.begin(), B.column_indices.begin()));

    cusp::array1d<IndexType,MemorySpace> indices(num_entries);
    thrust::sequence(exec, indices.begin(), indices.end());

    thrust::merge_by_key(exec,
                         A_tuples, A_tuples + A_nnz,
                         B_tuples, B_tuples + B_nnz,
                         thrust::counting_iterator<IndexType>(0),
                         thrust::counting_iterator<IndexType>(A_nnz),
                         thrust::make_discard_iterator(),
                         indices.begin(),
                         cusp::detail::coo_tuple_comp<IndexType>());

    JoinIndexIterator combined_tuples(A_tuples, A_tuples + A_nnz, B_tuples, B_tuples + B_nnz, indices.begin());

    TransValueIterator vals(B.values.begin(), UnaryOp());
    JoinValueIterator combined_values(A.values.begin(), A.values.begin() + A_nnz, vals, vals + B_nnz, indices.begin());

    // compute unique number of nonzeros in the output
    IndexType C_nnz = thrust::inner_product(exec,
                                            combined_tuples.begin(),
                                            combined_tuples.end() - 1,
                                            combined_tuples.begin() + 1,
                                            IndexType(1),
                                            thrust::plus<IndexType>(),
                                            thrust::not_equal_to< thrust::tuple<IndexType,IndexType> >());

    // allocate space for output
    C.resize(A.num_rows, A.num_cols, C_nnz);

    // sum values with the same (i,j)

    thrust::reduce_by_key(exec,
                          combined_tuples.begin(),
                          combined_tuples.end(),
                          combined_values.begin(),
                          thrust::make_zip_iterator(thrust::make_tuple(C.row_indices.begin(), C.column_indices.begin())),
                          C.values.begin(),
                          thrust::equal_to< thrust::tuple<IndexType,IndexType> >(),
                          BinaryOp());
#else
    cusp::array1d<IndexType,MemorySpace> rows(num_entries);
    cusp::array1d<IndexType,MemorySpace> cols(num_entries);
    cusp::array1d<ValueType,MemorySpace> vals(num_entries);

    thrust::copy(exec, A.row_indices.begin(),    A.row_indices.end(),    rows.begin());
    thrust::copy(exec, B.row_indices.begin(),    B.row_indices.end(),    rows.begin() + A_nnz);
    thrust::copy(exec, A.column_indices.begin(), A.column_indices.end(), cols.begin());
    thrust::copy(exec, B.column_indices.begin(), B.column_indices.end(), cols.begin() + A_nnz);
    thrust::copy(exec, A.values.begin(),         A.values.end(),         vals.begin());

    // apply transformation to B's values
    thrust::transform(exec, B.values.begin(), B.values.end(), vals.begin() + A_nnz, UnaryOp());

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
#endif

    int num_zeros = thrust::count(exec, C.values.begin(), C.values.end(), ValueType(0));

    // The result of the elementwise operation may contain zero entries so we need
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
          typename BinaryFunction,
          typename Format>
void elementwise(thrust::execution_policy<DerivedPolicy>& exec,
                 const MatrixType1& A, const MatrixType2& B, MatrixType3& C,
                 BinaryFunction op,
                 Format)
{
    typedef typename cusp::detail::coo_view_type<MatrixType1>::view View1;
    typedef typename cusp::detail::coo_view_type<MatrixType2>::view View2;
    typedef typename cusp::detail::as_coo_type<MatrixType3>::type CooMatrixType;

    View1 A_coo(A);
    View2 B_coo(B);
    CooMatrixType C_coo;

    cusp::elementwise(exec, A_coo, B_coo, C_coo, op);

    cusp::convert(C_coo, C);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp

