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
#include <cusp/format.h>
#include <cusp/coo_matrix.h>

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

    thrust::transform(A.values.values.begin(), A.values.values.end(),
                      B.values.values.begin(),
                      C.values.values.begin(),
                      op);
}

template <typename DerivedPolicy,
          typename MatrixType1, typename MatrixType2, typename MatrixType3,
          typename BinaryFunction>
void elementwise(thrust::execution_policy<DerivedPolicy>& exec,
                 const MatrixType1& A, const MatrixType2& B, MatrixType3& C,
                 BinaryFunction op,
                 cusp::coo_format& format)
{
    std::cout << " Performing elementwise COO " << std::endl;

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

    thrust::copy(A.row_indices.begin(),    A.row_indices.end(),    rows.begin());
    thrust::copy(B.row_indices.begin(),    B.row_indices.end(),    rows.begin() + A_nnz);
    thrust::copy(A.column_indices.begin(), A.column_indices.end(), cols.begin());
    thrust::copy(B.column_indices.begin(), B.column_indices.end(), cols.begin() + A_nnz);
    thrust::copy(A.values.begin(),         A.values.end(),         vals.begin());

    // apply transformation to B's values
    if(thrust::detail::is_same< BinaryFunction, thrust::minus<ValueType> >::value)
      thrust::transform(B.values.begin(), B.values.end(), vals.begin() + A_nnz, thrust::negate<ValueType>());

    // sort by (I,J)
    cusp::detail::sort_by_row_and_column(rows, cols, vals);

    // compute unique number of nonzeros in the output
    IndexType C_nnz = thrust::inner_product(thrust::make_zip_iterator(thrust::make_tuple(rows.begin(), cols.begin())),
                                            thrust::make_zip_iterator(thrust::make_tuple(rows.end (),  cols.end()))   - 1,
                                            thrust::make_zip_iterator(thrust::make_tuple(rows.begin(), cols.begin())) + 1,
                                            IndexType(1),
                                            thrust::plus<IndexType>(),
                                            thrust::not_equal_to< thrust::tuple<IndexType,IndexType> >());

    // allocate space for output
    C.resize(A.num_rows, A.num_cols, C_nnz);

    // sum values with the same (i,j)
    thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(rows.begin(), cols.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(rows.end(),   cols.end())),
                          vals.begin(),
                          thrust::make_zip_iterator(thrust::make_tuple(C.row_indices.begin(), C.column_indices.begin())),
                          C.values.begin(),
                          thrust::equal_to< thrust::tuple<IndexType,IndexType> >(),
                          thrust::plus<ValueType>());
}

template <typename DerivedPolicy,
          typename MatrixType1, typename MatrixType2, typename MatrixType3,
          typename BinaryFunction>
void elementwise(thrust::execution_policy<DerivedPolicy>& exec,
                 const MatrixType1& A, const MatrixType2& B, MatrixType3& C,
                 BinaryFunction op, sparse_format& format)
{
    std::cout << " Converting for elementwise " << std::endl;

    typedef typename MatrixType1::index_type   IndexType1;
    typedef typename MatrixType1::value_type   ValueType1;
    typedef typename MatrixType1::memory_space MemorySpace1;

    typedef typename MatrixType2::index_type   IndexType2;
    typedef typename MatrixType2::value_type   ValueType2;
    typedef typename MatrixType2::memory_space MemorySpace2;

    typedef typename MatrixType3::index_type   IndexType3;
    typedef typename MatrixType3::value_type   ValueType3;
    typedef typename MatrixType3::memory_space MemorySpace3;

    cusp::coo_matrix<IndexType1,ValueType1,MemorySpace1> A_coo(A);
    cusp::coo_matrix<IndexType2,ValueType2,MemorySpace2> B_coo(B);
    cusp::coo_matrix<IndexType3,ValueType3,MemorySpace3> C_coo;

    cusp::elementwise(exec, A_coo, B_coo, C_coo, op, coo_format());

    cusp::convert(C_coo, C);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp
