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

#include <cusp/detail/config.h>
#include <cusp/exception.h>
#include <cusp/multiply.h>

#include <thrust/execution_policy.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace generic
{
namespace detail
{

struct process_mis_nodes
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        if (thrust::get<1>(t) == 1)                     // undecided node
        {
            if (thrust::get<0>(t) == thrust::get<3>(t)) // i == maximal_index
                thrust::get<1>(t) = 2;                    // mis_node
        }
    }
};

struct process_non_mis_nodes
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        if (thrust::get<0>(t) == 1)            // undecided node
        {
            if (thrust::get<1>(t) == 2)        // maximal_state == mis_node
                thrust::get<0>(t) = 0;           // non_mis_node
        }
    }
};

template <typename Array1,
         typename Array2,
         typename Array3,
         typename Array4>
void compute_mis_states(const size_t k,
                        const Array1& row_indices,
                        const Array2& column_indices,
                        Array3& random_values,
                        Array4& states)
{
    typedef typename Array1::value_type   IndexType;
    typedef typename Array3::value_type   RandomType;
    typedef typename Array4::value_type   NodeStateType;
    typedef typename Array1::memory_space MemorySpace;

    typedef typename thrust::counting_iterator<IndexType>                     CountingIterator;
    typedef typename cusp::array1d<NodeStateType,MemorySpace>::iterator       StatesIterator;
    typedef typename cusp::array1d<RandomType,MemorySpace>::iterator          RandomIterator;
    typedef typename cusp::array1d<IndexType,MemorySpace>::iterator           IndexIterator;

    typedef typename thrust::tuple<NodeStateType,RandomType,IndexType>        Tuple1;
    typedef thrust::tuple<StatesIterator,RandomIterator,CountingIterator>     IteratorTuple1;
    typedef typename thrust::zip_iterator<IteratorTuple1>                     ZipIterator1;
    typedef typename cusp::array1d_view<ZipIterator1>                         ArrayType1;

    typedef typename thrust::tuple<NodeStateType,RandomType,IndexType>        Tuple2;
    typedef thrust::tuple<StatesIterator,RandomIterator,IndexIterator>        IteratorTuple2;
    typedef typename thrust::zip_iterator<IteratorTuple2>                     ZipIterator2;
    typedef typename cusp::array1d_view<ZipIterator2>                         ArrayType2;

    typedef typename Array1::const_view                                       RowView;
    typedef typename Array2::const_view                                       ColumnView;
    typedef typename cusp::constant_array<Tuple1>                             ValueView;
    typedef typename cusp::coo_matrix_view<RowView,ColumnView,ValueView>      CooView;

    const size_t N = states.size();
    const size_t M = row_indices.size();

    cusp::array1d<NodeStateType,MemorySpace> maximal_states(N);
    cusp::array1d<RandomType,MemorySpace>    maximal_values(N);
    cusp::array1d<IndexType,MemorySpace>     maximal_indices(N);

    cusp::array1d<NodeStateType,MemorySpace> last_states;
    cusp::array1d<RandomType,MemorySpace>    last_values;
    cusp::array1d<IndexType,MemorySpace>     last_indices;

    if(k >= 1)
    {
        last_states.resize (N);
        last_values.resize (N);
        last_indices.resize(N);
    }

    cusp::constant_array<Tuple1> values(M, Tuple1(0,0,0));
    CooView A(N, N, M, make_array1d_view(row_indices), make_array1d_view(column_indices), values);

    CountingIterator count_begin(0);
    ZipIterator1 x_iter(thrust::make_tuple(states.begin(), random_values.begin(), count_begin));
    ZipIterator2 y_iter(thrust::make_tuple(last_states.begin(), last_values.begin(), last_indices.begin()));
    ZipIterator2 z_iter(thrust::make_tuple(maximal_states.begin(), maximal_values.begin(), maximal_indices.begin()));

    ArrayType1 x(x_iter, x_iter + N);
    ArrayType2 y(y_iter, y_iter + N);
    ArrayType2 z(z_iter, z_iter + N);

    size_t active_nodes = N;

    do
    {
        // find the largest (state,value,index) 1-ring neighbor for each node
        cusp::generalized_spmv(A, x, x, z, thrust::project2nd<Tuple1,Tuple2>(), thrust::maximum<Tuple2>());

        // find the largest (state,value,index) k-ring neighbor for each node (if k > 1)
        for(size_t ring = 1; ring < k; ring++)
        {
            last_states.swap (maximal_states);
            last_indices.swap(maximal_indices);
            last_values.swap (maximal_values);
            y.swap(z);

            cusp::generalized_spmv(A, y, y, z, thrust::project2nd<Tuple1,Tuple2>(), thrust::maximum<Tuple2>());
        }

        // label local maxima as MIS nodes
        thrust::for_each(thrust::make_zip_iterator(
                             thrust::make_tuple(
                               thrust::counting_iterator<IndexType>(0), states.begin(),
                               maximal_states.begin(), maximal_indices.begin())),
                         thrust::make_zip_iterator(
                             thrust::make_tuple(
                               thrust::counting_iterator<IndexType>(0), states.begin(),
                               maximal_states.begin(), maximal_indices.begin())) + N,
                         process_mis_nodes());

        // label k-ring neighbors of MIS nodes as non-MIS nodes
        thrust::for_each(thrust::make_zip_iterator(
                             thrust::make_tuple(states.begin(), thrust::make_permutation_iterator(states.begin(), maximal_indices.begin()))),
                         thrust::make_zip_iterator(
                             thrust::make_tuple(states.begin(), thrust::make_permutation_iterator(states.begin(), maximal_indices.begin()))) + N,
                         process_non_mis_nodes());

        active_nodes = thrust::count(states.begin(), states.end(), 1);

    } while (active_nodes > 0);
}

} // end namespace detail

template <typename DerivedPolicy, typename MatrixType, typename ArrayType>
size_t maximal_independent_set(thrust::execution_policy<DerivedPolicy>& exec,
                               const MatrixType& G,
                               ArrayType& stencil,
                               const size_t k,
                               coo_format)
{
    typedef typename MatrixType::index_type   IndexType;
    typedef typename MatrixType::value_type   ValueType;
    typedef typename MatrixType::memory_space MemorySpace;
    typedef unsigned int  RandomType;
    typedef unsigned char NodeStateType;

    const IndexType N = G.num_rows;

    cusp::array1d<RandomType,MemorySpace> random_values(N);
    cusp::copy(cusp::random_array<RandomType>(N), random_values);

    cusp::array1d<NodeStateType,MemorySpace> states(N, 1);

    detail::compute_mis_states(k, G.row_indices, G.column_indices, random_values, states);

    // resize output
    stencil.resize(N);

    // mark all mis nodes
    thrust::transform(states.begin(), states.end(), thrust::constant_iterator<NodeStateType>(2), stencil.begin(), thrust::equal_to<NodeStateType>());

    // return the size of the MIS
    return thrust::count(stencil.begin(), stencil.end(), typename ArrayType::value_type(true));
}

template <typename DerivedPolicy, typename MatrixType, typename ArrayType>
size_t maximal_independent_set(thrust::execution_policy<DerivedPolicy>& exec,
                               const MatrixType& G,
                               ArrayType& stencil,
                               const size_t k,
                               csr_format)
{
    typedef typename MatrixType::column_indices_array_type     IndicesType;
    typedef typename IndicesType::view                         IndicesView;
    typedef typename IndicesType::const_view                   ConstIndicesView;
    typedef typename MatrixType::values_array_type::const_view ConstValuesView;

    IndicesType row_indices(G.num_entries);
    cusp::offsets_to_indices(G.row_offsets, row_indices);

    cusp::coo_matrix_view<IndicesView,ConstIndicesView,ConstValuesView> G_coo(
        G.num_rows, G.num_cols, G.num_entries,
        cusp::make_array1d_view(row_indices),
        cusp::make_array1d_view(G.column_indices),
        cusp::make_array1d_view(G.values));

    return cusp::graph::maximal_independent_set(exec, G_coo, stencil, k);
}

template <typename DerivedPolicy, typename MatrixType, typename ArrayType>
size_t maximal_independent_set(thrust::execution_policy<DerivedPolicy>& exec,
                               const MatrixType& G,
                               ArrayType& stencil,
                               const size_t k,
                               known_format)
{
    typedef typename cusp::detail::as_csr_type<MatrixType>::type CsrMatrix;

    CsrMatrix G_csr(G);

    return cusp::graph::maximal_independent_set(exec, G_csr, stencil, k);
}

} // end namespace generic
} // end namespace detail
} // end namespace system
} // end namespace cusp
