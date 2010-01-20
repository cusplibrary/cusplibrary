#include <cusp/detail/device/generalized_spmv/csr_scalar.h>

#include <cusp/csr_matrix.h>
#include <cusp/array1d.h>
#include <cusp/exception.h>

#include <thrust/count.h>
#include <thrust/transform.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

namespace cusp
{
namespace graph
{
namespace detail
{

// http://burtleburtle.net/bob/hash/integer.html
struct simple_hash
{
    unsigned int operator()(unsigned int a)
    {
        a = (a + 0x7ed55d16) + (a << 12);
        a = (a ^ 0xc761c23c) ^ (a >> 19);
        a = (a + 0x165667b1) + (a <<  5);
        a = (a + 0xd3a2646c) ^ (a <<  9);
        a = (a + 0xfd7046c5) + (a <<  3);
        a = (a ^ 0xb55a4f09) ^ (a >> 16);
        return a;
    }
};

struct process_nodes
{
    template <typename Tuple>
    void operator()(Tuple t)
    {
        if (thrust::get<1>(t) == 1)                     // undecided node
        {
            if (thrust::get<0>(t) == thrust::get<3>(t)) // i == maximal_index
              thrust::get<1>(t) = 2;                    // mis_node
            else if (thrust::get<2>(t) == 2)            // maximal_state == mis_node
              thrust::get<1>(t) = 0;                    // non_mis_node
        }
    }
};

template <typename IndexType, typename ValueType, typename SpaceOrAlloc, typename ArrayType>
size_t maximal_independent_set(const cusp::csr_matrix<IndexType,ValueType,SpaceOrAlloc>& A, ArrayType& stencil)
{
    typedef unsigned int RandomType;
    typedef unsigned int NodeStateType;
        
    if(A.num_rows != A.num_cols)
        throw cusp::invalid_input_exception("matrix must be square");
    
    // throw if A.num_rows != A.num_cols
    const IndexType N = A.num_rows;

    cusp::array1d<RandomType,SpaceOrAlloc> random_values(N);
    thrust::transform(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(N), 
                      random_values.begin(),
                      simple_hash());

    cusp::array1d<NodeStateType,SpaceOrAlloc> states(N, 1);
    
    typedef typename thrust::tuple<NodeStateType,RandomType,IndexType> Tuple;
    
    cusp::array1d<NodeStateType,SpaceOrAlloc> maximal_states(N);
    cusp::array1d<RandomType,SpaceOrAlloc>    maximal_values(N);
    cusp::array1d<IndexType,SpaceOrAlloc>     maximal_indices(N);

    size_t num_iterations = 0;

    while (thrust::count(states.begin(), states.end(), 1) > 0)
    {
        // find the largest (state,value,index) neighbor for each node
        cusp::detail::device::cuda::spmv_csr_scalar
            (A.num_rows,
             A.row_offsets.begin(), A.column_indices.begin(), thrust::constant_iterator<int>(1),  // XXX should we mask explicit zeros? (e.g. DIA, array2d)
             thrust::make_zip_iterator(thrust::make_tuple(states.begin(), random_values.begin(), thrust::counting_iterator<IndexType>(0))),
             thrust::make_zip_iterator(thrust::make_tuple(states.begin(), random_values.begin(), thrust::counting_iterator<IndexType>(0))),
             thrust::make_zip_iterator(thrust::make_tuple(maximal_states.begin(), maximal_values.begin(), maximal_indices.begin())),
             thrust::identity<Tuple>(), thrust::project2nd<Tuple,Tuple>(), thrust::maximum<Tuple>());

        // label local maxima as MIS nodes and neighbors of MIS nodes as non-MIS nodes
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<IndexType>(0), states.begin(), maximal_states.begin(), maximal_indices.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<IndexType>(0), states.begin(), maximal_states.begin(), maximal_indices.begin())) + N,
                         process_nodes());

        num_iterations++;
    }

    // mark all mis nodes
    thrust::transform(states.begin(), states.end(), thrust::constant_iterator<NodeStateType>(2), stencil.begin(), thrust::equal_to<NodeStateType>());

    return num_iterations;
}

} // end namespace detail

template <typename MatrixType, typename ArrayType>
size_t maximal_independent_set(MatrixType& A, ArrayType& stencil)
{
    return cusp::graph::detail::maximal_independent_set(A, stencil);
}

} // end namespace graph
} // end namespace cusp

