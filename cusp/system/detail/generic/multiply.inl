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

template<typename BinaryFunction>
struct combine_functor : public BinaryFunction
{
    typedef typename BinaryFunction::result_type T;

    template<typename Tuple>
    __host__ __device__
    T operator()(const Tuple& t)
    {
        return BinaryFunction::operator()(thrust::get<0>(t), thrust::get<1>(t));
    }
};

template<typename T>
struct max_functor : public thrust::binary_function<T,T,T>
{
    const T base;

    max_functor(const T base)
      : base(base) {}

    __host__ __device__
    T operator()(const T col)
    {
        return max(base, col);
    }
};

template <typename T>
struct modulus_functor : public thrust::binary_function<T,T,T>
{
    const T base;

    modulus_functor(const T base)
        : base(base) {}

    __host__ __device__
    T operator()(const T& x) const
    {
        return x % base;
    }
};

template <typename T>
struct divide_functor : public thrust::binary_function<T,T,T>
{
    const T base;

    divide_functor(const T base)
        : base(base) {}

    __host__ __device__
    T operator()(const T& x) const
    {
        return x / base;
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
              dia_format&, array1d_format&, array1d_format&)
{
    typedef typename LinearOperator::index_type IndexType;
    typedef typename LinearOperator::value_type ValueType;

    // define types used to programatically generate row_indices
    typedef typename thrust::counting_iterator<IndexType> IndexIterator;
    typedef typename thrust::transform_iterator<modulus_functor<IndexType>, IndexIterator> RowIndexIterator;

    RowIndexIterator row_indices_begin(IndexIterator(0), modulus_functor<IndexType>(A.values.pitch));

    // define types used to programatically generate column_indices
    typedef combine_functor< thrust::plus<IndexType> > sum_tuple_functor;
    typedef typename thrust::device_vector<IndexType>::const_iterator ConstElementIterator;
    typedef typename thrust::transform_iterator<divide_functor<IndexType>, IndexIterator> DivideIterator;
    typedef typename thrust::permutation_iterator<ConstElementIterator,DivideIterator> OffsetsPermIterator;
    typedef typename thrust::tuple<OffsetsPermIterator, RowIndexIterator> IteratorTuple;
    typedef typename thrust::zip_iterator<IteratorTuple> ZipIterator;
    typedef typename thrust::transform_iterator<sum_tuple_functor, ZipIterator> ColumnIndexIterator;

    DivideIterator gather_indices_begin(IndexIterator(0), divide_functor<IndexType>(A.values.pitch));
    OffsetsPermIterator offsets_begin(A.diagonal_offsets.begin(), gather_indices_begin);
    ZipIterator offset_modulus_tuple(thrust::make_tuple(offsets_begin, row_indices_begin));
    ColumnIndexIterator column_indices_begin(offset_modulus_tuple, sum_tuple_functor());

    thrust::detail::temporary_array<ValueType, DerivedPolicy> vals(exec, A.num_rows);

    thrust::reduce_by_key(exec,
                          row_indices_begin, row_indices_begin + A.values.values.size(),
                          thrust::make_transform_iterator(
                              thrust::make_zip_iterator(
                                  thrust::make_tuple(
                                    thrust::make_permutation_iterator(B.begin(),
                                      thrust::make_transform_iterator(column_indices_begin, max_functor<IndexType>(0))),
                                      A.values.values.begin())),
                              combine_functor<BinaryFunction1>()),
                          thrust::make_discard_iterator(),
                          vals.begin(),
                          thrust::equal_to<IndexType>(),
                          reduce);

    thrust::transform(exec, vals.begin(), vals.end(), thrust::make_transform_iterator(C.begin(), initialize), C.begin(), reduce);
}

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
              ell_format&, array1d_format&, array1d_format&)
{
    typedef typename LinearOperator::index_type IndexType;
    typedef typename LinearOperator::value_type ValueType;

    // define types used to programatically generate row_indices
    typedef typename thrust::counting_iterator<IndexType> IndexIterator;
    typedef typename thrust::transform_iterator<modulus_functor<IndexType>, IndexIterator> RowIndexIterator;

    RowIndexIterator row_indices_begin(IndexIterator(0), modulus_functor<IndexType>(A.values.pitch));

    thrust::detail::temporary_array<ValueType, DerivedPolicy> vals(exec, A.num_rows);

    thrust::reduce_by_key(exec,
                          row_indices_begin, row_indices_begin + A.values.values.size(),
                          thrust::make_transform_iterator(
                              thrust::make_zip_iterator(
                                  thrust::make_tuple(
                                    thrust::make_permutation_iterator(B.begin(),
                                      thrust::make_transform_iterator(A.column_indices.values.begin(), max_functor<IndexType>(0))),
                                      A.values.values.begin())),
                              combine_functor<BinaryFunction1>()),
                          thrust::make_discard_iterator(),
                          vals.begin(),
                          thrust::equal_to<IndexType>(),
                          reduce);

    thrust::transform(exec, vals.begin(), vals.end(), thrust::make_transform_iterator(C.begin(), initialize), C.begin(), reduce);
}

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
    typedef typename LinearOperator::value_type ValueType;
    typedef typename LinearOperator::memory_space MemorySpace;

    thrust::detail::temporary_array<IndexType, DerivedPolicy> rows(exec, A.num_rows);
    thrust::detail::temporary_array<ValueType, DerivedPolicy> vals(exec, A.num_rows);

    typename thrust::detail::temporary_array<IndexType, DerivedPolicy>::iterator rows_end;
    typename thrust::detail::temporary_array<ValueType, DerivedPolicy>::iterator vals_end;

    thrust::tie(rows_end, vals_end) =
        thrust::reduce_by_key(exec,
                              A.row_indices.begin(), A.row_indices.end(),
                              thrust::make_transform_iterator(
                                  thrust::make_zip_iterator(
                                      thrust::make_tuple(
                                        thrust::make_permutation_iterator(B.begin(),
                                          thrust::make_transform_iterator(A.column_indices.begin(), max_functor<IndexType>(0))),
                                          A.values.begin())),
                                  combine_functor<BinaryFunction1>()),
                              rows.begin(),
                              vals.begin(),
                              thrust::equal_to<IndexType>(),
                              reduce);

    thrust::transform(C.begin(), C.end(), C.begin(), initialize);

    int num_entries = rows_end - rows.begin();
    thrust::transform(exec,
                      thrust::make_permutation_iterator(C.begin(), rows.begin()),
                      thrust::make_permutation_iterator(C.begin(), rows.begin()) + num_entries,
                      vals.begin(),
                      thrust::make_permutation_iterator(C.begin(), rows.begin()),
                      reduce);
}

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
              csr_format&,
              array1d_format&,
              array1d_format&)
{
    typedef typename LinearOperator::index_type IndexType;
    typedef typename thrust::detail::temporary_array<IndexType, DerivedPolicy>::iterator TempIterator;

    typedef typename cusp::array1d_view<TempIterator>::view           RowView;
    typedef typename LinearOperator::column_indices_array_type::view  ColView;
    typedef typename LinearOperator::values_array_type::view          ValView;

    thrust::detail::temporary_array<IndexType, DerivedPolicy> temp(exec, A.num_entries);
    cusp::array1d_view<TempIterator> row_indices(temp.begin(), temp.end());
    cusp::detail::offsets_to_indices(A.row_offsets, row_indices);

    cusp::coo_matrix_view<RowView,ColView,ValView> A_coo_view(A.num_rows, A.num_cols, A.num_entries,
                                                              cusp::make_array1d_view(row_indices),
                                                              cusp::make_array1d_view(A.column_indices),
                                                              cusp::make_array1d_view(A.values));

    cusp::multiply(exec, A_coo_view, B, C, initialize, combine, reduce);
}

template <typename DerivedPolicy,
         typename LinearOperator, typename MatrixOrVector1, typename MatrixOrVector2,
         typename UnaryFunction,  typename BinaryFunction1, typename BinaryFunction2>
void multiply(thrust::execution_policy<DerivedPolicy> &exec,
              LinearOperator&  A,
              MatrixOrVector1& B,
              MatrixOrVector2& C,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce,
              cusp::permutation_format,
              cusp::array1d_format,
              cusp::array1d_format)
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
              UnaryFunction  initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce,
              hyb_format&,
              array1d_format&,
              array1d_format&)
{
    typedef typename MatrixOrVector2::value_type ValueType;

    cusp::multiply(exec, A.ell, B, C, initialize, combine, reduce);
    cusp::multiply(exec, A.coo, B, C, thrust::identity<ValueType>(), combine, reduce);
}

template <typename DerivedPolicy,
         typename LinearOperator,
         typename MatrixOrVector1,
         typename MatrixOrVector2>
typename thrust::detail::enable_if<!thrust::detail::is_convertible<typename LinearOperator::format,known_format>::value,void>::type
multiply(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
         const LinearOperator&  A,
         const MatrixOrVector1& B,
         MatrixOrVector2& C)
{
    // user-defined LinearOperator
    ((LinearOperator&)A)(B,C);
}

template <typename DerivedPolicy,
         typename LinearOperator,
         typename MatrixOrVector1,
         typename MatrixOrVector2>
typename thrust::detail::enable_if<thrust::detail::is_convertible<typename LinearOperator::format,known_format>::value,void>::type
multiply(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
         const LinearOperator&  A,
         const MatrixOrVector1& B,
         MatrixOrVector2& C)
{
    typedef typename LinearOperator::value_type ValueType;

    cusp::detail::zero_function<ValueType> initialize;
    thrust::multiplies<ValueType> combine;
    thrust::plus<ValueType> reduce;

    cusp::multiply(exec, A, B, C, initialize, combine, reduce);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp
