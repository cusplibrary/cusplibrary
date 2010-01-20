#include <unittest/unittest.h>

#include <cusp/detail/device/generalized_spmv/csr_scalar.h>

#include <cusp/csr_matrix.h>
#include <cusp/array1d.h>
#include <cusp/gallery/poisson.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <cusp/print.h> // TODO remove

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
void maximal_independent_set(const cusp::csr_matrix<IndexType,ValueType,SpaceOrAlloc>& A, ArrayType& stencil)
{
    typedef unsigned int RandomType;
    typedef unsigned int NodeStateType;
    
    // throw if A.num_rows != A.num_cols
    const IndexType N = A.num_rows;

    cusp::array1d<RandomType,SpaceOrAlloc> random_values(N);
    thrust::transform(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(N), 
                      random_values.begin(),
                      simple_hash());

    //std::cout << "random_values\n";
    //cusp::print_matrix(random_values);

    cusp::array1d<NodeStateType,SpaceOrAlloc> states(N, 1);
    
    //std::cout << "states\n";
    //cusp::print_matrix(states);

    typedef typename thrust::tuple<NodeStateType,RandomType,IndexType> Tuple;
    
    cusp::array1d<NodeStateType,SpaceOrAlloc> maximal_states(N);
    cusp::array1d<RandomType,SpaceOrAlloc>    maximal_values(N);
    cusp::array1d<IndexType,SpaceOrAlloc>     maximal_indices(N);

    while (thrust::count(states.begin(), states.end(), 1) > 0)
    {
        cusp::detail::device::cuda::spmv_csr_scalar
            (A.num_rows,
             A.row_offsets.begin(), A.column_indices.begin(), A.values.begin(),  // values array is irrelevant
             thrust::make_zip_iterator(thrust::make_tuple(states.begin(), random_values.begin(), thrust::counting_iterator<IndexType>(0))),
             thrust::make_zip_iterator(thrust::make_tuple(states.begin(), random_values.begin(), thrust::counting_iterator<IndexType>(0))),
             thrust::make_zip_iterator(thrust::make_tuple(maximal_states.begin(), maximal_values.begin(), maximal_indices.begin())),
             thrust::identity<Tuple>(), thrust::project2nd<Tuple,Tuple>(), thrust::maximum<Tuple>());

        //std::cout << "maximal_states: ";
        //cusp::print_matrix(maximal_states);
        //std::cout << "maximal_values: ";
        //cusp::print_matrix(maximal_values);
        //std::cout << "maximal_indices: ";
        //cusp::print_matrix(maximal_indices);

        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<IndexType>(0), states.begin(), maximal_states.begin(), maximal_indices.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<IndexType>(0), states.begin(), maximal_states.begin(), maximal_indices.begin())) + N,
                         process_nodes());

        //std::cout << "states: ";
        //cusp::print_matrix(states);
        //std::cout << "=======================================================\n";
    }

    // mark all mis nodes
    thrust::transform(states.begin(), states.end(), thrust::constant_iterator<NodeStateType>(2), stencil.begin(), thrust::equal_to<NodeStateType>());
}

template <typename IndexType, typename ValueType, typename SpaceOrAlloc, typename ArrayType>
bool is_valid_mis(const cusp::csr_matrix<IndexType,ValueType,SpaceOrAlloc>& A, ArrayType& stencil)
{
    cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> csr(A);
    cusp::array1d<int,cusp::host_memory> mis(stencil);

    // check that e
    for (IndexType i = 0; i < csr.num_rows; i++)
    {
        IndexType num_mis_neighbors = 0;
        for(IndexType jj = csr.row_offsets[i]; jj < csr.row_offsets[i + 1]; jj++)
        {
            IndexType j = A.column_indices[jj];
            
            if (i != j && mis[j])
                num_mis_neighbors++;
        }

        if (mis[i])
        {
            if(num_mis_neighbors > 0)
            {
                std::cout << "Node " << i << " conflicts with another node" << std::endl;
                return false;
            }
        }
        else
        {
            if (num_mis_neighbors == 0)
            {
               std::cout << "Node " << i << " is not in the MIS and has no MIS neighbors" << std::endl;
               return false; 
            }
        }
    }

    return true;
}

void TestCsrMaximalIndependentSet(void)
{
    typedef cusp::csr_matrix<int,float,cusp::device_memory> TestMatrix;
    typedef typename TestMatrix::memory_space MemorySpace;

    // initialize test matrix
    TestMatrix A;
    cusp::gallery::poisson5pt(A, 50, 13);

    // allocate storage for MIS result
    cusp::array1d<int, MemorySpace> stencil(A.num_rows, 0);

    // compute MIS
    maximal_independent_set(A, stencil);

    // check MIS
    is_valid_mis(A, stencil);
}
DECLARE_UNITTEST(TestCsrMaximalIndependentSet);

