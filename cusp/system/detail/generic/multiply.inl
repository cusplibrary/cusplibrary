/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

#include <thrust/detail/config.h>
#include <thrust/reduce.h>

#include <thrust/system/detail/generic/tag.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cusp/format.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/print.h>

#include <cusp/detail/utils.h>
#include <cusp/detail/array2d_format_utils.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{

template <typename DerivedPolicy,
         typename LinearOperator, typename MatrixOrVector1, typename MatrixOrVector2,
         typename UnaryFunction,  typename BinaryFunction1, typename BinaryFunction2>
void multiply(thrust::execution_policy<DerivedPolicy> &exec,
              LinearOperator&  A,
              MatrixOrVector1& B,
              MatrixOrVector2& C,
              UnaryFunction  initialize, BinaryFunction1 combine, BinaryFunction2 reduce,
              cusp::permutation_format, cusp::array1d_format, cusp::array1d_format)
{
    // TODO : initialize, combine, and reduce are ignored
    thrust::gather(A.permutation.begin(), A.permutation.end(), B.begin(), C.begin());
}

template <typename DerivedPolicy,
         typename LinearOperator, typename MatrixOrVector1, typename MatrixOrVector2,
         typename UnaryFunction,  typename BinaryFunction1, typename BinaryFunction2>
void multiply(thrust::execution_policy<DerivedPolicy> &exec,
              LinearOperator&  A,
              MatrixOrVector1& B,
              MatrixOrVector2& C,
              UnaryFunction  initialize, BinaryFunction1 combine, BinaryFunction2 reduce,
              hyb_format&, array1d_format&, array1d_format&)
{
    typedef typename MatrixOrVector2::value_type ValueType;

    cusp::multiply(exec, A.ell, B, C, initialize, combine, reduce);
    cusp::multiply(exec, A.coo, B, C, thrust::identity<ValueType>(), combine, reduce);
}

template<typename BinaryFunction>
struct spmv_struct
{
    typedef typename BinaryFunction::result_type result_type;
    BinaryFunction combine;

    template<typename Tuple>
    __host__ __device__
    result_type operator()(const Tuple& t)
    {
        return combine(thrust::get<0>(t), thrust::get<1>(t));
    }
};

template <typename DerivedPolicy,
         typename LinearOperator, typename MatrixOrVector1, typename MatrixOrVector2,
         typename UnaryFunction,  typename BinaryFunction1, typename BinaryFunction2>
void multiply(thrust::execution_policy<DerivedPolicy> &exec,
              LinearOperator&  A,
              MatrixOrVector1& B,
              MatrixOrVector2& C,
              UnaryFunction    initialize,
              BinaryFunction1  combine,
              BinaryFunction2  reduce,
              coo_format&, array1d_format&, array1d_format&)
{
    typedef typename LinearOperator::index_type IndexType;
    typedef typename LinearOperator::memory_space MemorySpace;

    typedef cusp::array1d<IndexType,MemorySpace> IndexArray;
    typedef typename IndexArray::iterator        IndexIterator;
    typedef typename MatrixOrVector2::iterator   ValueIterator;

    IndexIterator index_last;
    ValueIterator value_last;

    IndexArray rows(A.num_rows);
    MatrixOrVector2 vals(A.num_rows);

    thrust::tie(index_last, value_last) =
        thrust::reduce_by_key(exec,
                              A.row_indices.begin(), A.row_indices.end(),
                              thrust::make_transform_iterator(
                                  thrust::make_zip_iterator(
                                      thrust::make_tuple(thrust::make_permutation_iterator(B.begin(), A.column_indices.begin()), A.values.begin())),
                                  spmv_struct<BinaryFunction1>()),
                              rows.begin(),
                              vals.begin(),
                              thrust::equal_to<IndexType>(),
                              reduce);

    int num_entries = index_last - rows.begin();

    thrust::transform(exec, C.begin(), C.end(), C.begin(), initialize);

    thrust::transform(exec,
                      thrust::make_permutation_iterator(C.begin(), rows.begin()),
                      thrust::make_permutation_iterator(C.begin(), rows.begin()) + num_entries,
                      vals.begin(),
                      thrust::make_permutation_iterator(C.begin(), rows.begin()),
                      reduce);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp
