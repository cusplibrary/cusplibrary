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

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>

#include <cusp/multiply.h>
#include <cusp/graph/maximal_independent_set.h>

#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{
namespace detail
{

template <typename MatrixType, typename ArrayType>
void mis_to_aggregates(const MatrixType& C, const ArrayType& mis, ArrayType& aggregates)
{
    typedef typename MatrixType::index_type                                   IndexType;
    typedef typename MatrixType::value_type                                   ValueType;
    typedef typename MatrixType::memory_space                                 MemorySpace;

    typedef typename ArrayType::value_type                                    T;
    typedef typename ArrayType::iterator                                      ArrayIterator;
    typedef typename ArrayType::const_iterator                                ConstArrayIterator;

    typedef thrust::tuple<T,T>                                                Tuple;
    typedef typename thrust::counting_iterator<IndexType>                     CountingIterator;

    typedef typename MatrixType::row_indices_array_type::const_view           RowView;
    typedef typename MatrixType::column_indices_array_type::const_view        ColumnView;
    typedef typename cusp::constant_array<Tuple>                              ValueView;
    typedef typename cusp::coo_matrix_view<RowView,ColumnView,ValueView>      CooView;

    typedef thrust::tuple<ConstArrayIterator,CountingIterator>                IteratorTuple1;
    typedef typename thrust::zip_iterator<IteratorTuple1>                     ZipIterator1;
    typedef typename cusp::array1d_view<ZipIterator1>                         ArrayViewType1;

    typedef thrust::tuple<ArrayIterator,ArrayIterator>                        IteratorTuple2;
    typedef typename thrust::zip_iterator<IteratorTuple2>                     ZipIterator2;
    typedef typename cusp::array1d_view<ZipIterator2>                         ArrayViewType2;

    const IndexType N = C.num_rows;
    const IndexType M = C.num_entries;
    cusp::constant_array<Tuple> values(M, Tuple(1,1));
    CooView A(N, N, M, make_array1d_view(C.row_indices), make_array1d_view(C.column_indices), values);

    // current (ring,index)
    ArrayType mis1(N);
    ArrayType mis2(N);

    ArrayType idx1(N);
    ArrayType idx2(N);

    CountingIterator count_begin(0);
    ZipIterator1 x_iter(thrust::make_tuple(mis.begin(),  count_begin));
    ZipIterator2 y_iter(thrust::make_tuple(mis1.begin(), idx1.begin()));
    ZipIterator2 z_iter(thrust::make_tuple(mis2.begin(), idx2.begin()));

    ArrayViewType1 x(x_iter, x_iter + N);
    ArrayViewType2 y(y_iter, y_iter + N);
    ArrayViewType2 z(z_iter, z_iter + N);

    // (2,i) mis (0,i) non-mis
    // find the largest (mis[j],j) 1-ring neighbor for each node
    cusp::generalized_spmv(A, x, x, y, thrust::project2nd<Tuple,Tuple>(), thrust::maximum<Tuple>());

    // boost mis0 values so they win in second round
    thrust::transform(mis.begin(), mis.end(), mis1.begin(), mis1.begin(), thrust::plus<T>());

    // find the largest (mis[j],j) 2-ring neighbor for each node
    cusp::generalized_spmv(A, y, y, z, thrust::project2nd<Tuple,Tuple>(), thrust::maximum<Tuple>());

    // enumerate the MIS nodes
    cusp::array1d<IndexType,MemorySpace> mis_enum(N);
    thrust::exclusive_scan(mis.begin(), mis.end(), mis_enum.begin());

    thrust::gather(idx2.begin(), idx2.end(), mis_enum.begin(), aggregates.begin());
} // mis_to_aggregates()

template <typename DerivedPolicy, typename Matrix, typename Array>
void mis_aggregate(thrust::execution_policy<DerivedPolicy> &exec,
                   const Matrix& C, Array& aggregates, Array& mis,
                   cusp::coo_format)
{
    typedef typename Matrix::index_type IndexType;

    // compute MIS(2)
    cusp::graph::maximal_independent_set(C, mis, 2);

    // compute aggregates from MIS(2)
    mis_to_aggregates(C, mis, aggregates);

    // locate singletons
    IndexType num_aggregates = *thrust::max_element(aggregates.begin(), aggregates.end()) + 1;
    Array sorted_aggregates(aggregates);
    Array aggregate_counts(num_aggregates);
    Array reduced_aggregates(num_aggregates);

    // compute sizes of the aggregates
    thrust::sort(sorted_aggregates.begin(), sorted_aggregates.end());
    thrust::reduce_by_key(sorted_aggregates.begin(), sorted_aggregates.end(),
                          thrust::constant_iterator<IndexType>(1),
                          thrust::make_discard_iterator(),
                          aggregate_counts.begin());

    // count the number of aggregates consisting of a single node
    IndexType num_singletons = thrust::count(aggregate_counts.begin(), aggregate_counts.end(), IndexType(1));

    // mark singletons with -1 for filtering, the total number of aggregates is now (num_aggregates - num_singletons)
    if ( num_singletons > 0 ) {
        Array aggregate_ids(num_aggregates);
        cusp::array1d<bool,cusp::device_memory> isone(num_aggregates);

        // [2, 2, 1, 2, 2, 1] -> [1, 1, 0, 1, 1, 0]
        thrust::transform(aggregate_counts.begin(), aggregate_counts.end(),
                          thrust::constant_iterator<IndexType>(1), isone.begin(),
                          thrust::equal_to<IndexType>());
        // [1, 1, 0, 1, 1, 0] -> [0, 1, 2, 2, 3, 3]
        thrust::exclusive_scan(thrust::make_transform_iterator(isone.begin(), thrust::logical_not<bool>()),
                               thrust::make_transform_iterator(isone.end()  , thrust::logical_not<bool>()),
                               aggregate_ids.begin());
        // [0, 1, 2, 2, 3, 3] -> [0, 1, -1, 2, 3, -1]
        thrust::scatter_if( thrust::constant_iterator<IndexType>(-1),
                            thrust::constant_iterator<IndexType>(-1) + num_aggregates,
                            thrust::counting_iterator<IndexType>(0),
                            isone.begin(),
                            aggregate_ids.begin());
        thrust::gather(aggregates.begin(), aggregates.end(), aggregate_ids.begin(), aggregates.begin());
    }
}

template <typename DerivedPolicy, typename Matrix, typename Array>
void mis_aggregate(thrust::execution_policy<DerivedPolicy> &exec,
                   const Matrix& C, Array& aggregates, Array& mis,
                   cusp::known_format)
{
    typedef typename Matrix::const_coo_view_type MatrixViewType;

    MatrixViewType C_(C);

    mis_aggregate(exec, C_ , aggregates, mis, cusp::coo_format());
}

template <typename DerivedPolicy, typename Matrix, typename Array>
void mis_aggregate(thrust::execution_policy<DerivedPolicy> &exec,
                   const Matrix& C, Array& aggregates, Array& mis)
{
    typedef typename Matrix::format Format;

    Format format;

    mis_aggregate(exec, C, aggregates, mis, format);
}

} // end namespace detail
} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

