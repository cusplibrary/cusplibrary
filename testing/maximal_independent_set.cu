#include <unittest/unittest.h>

#include <cusp/graph/maximal_independent_set.h>

#include <cusp/gallery/poisson.h> 
#include <cusp/multiply.h>

// check whether the MIS is valid
template <typename MatrixType, typename ArrayType>
bool is_valid_mis(MatrixType& A, ArrayType& stencil)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;

    // convert matrix to CSR format on host
    cusp::csr_matrix<IndexType,ValueType,cusp::host_memory> csr(A);

    // copy mis array to host
    cusp::array1d<int,cusp::host_memory> mis(stencil);

    for (IndexType i = 0; i < csr.num_rows; i++)
    {
        IndexType num_mis_neighbors = 0;
        for(IndexType jj = csr.row_offsets[i]; jj < csr.row_offsets[i + 1]; jj++)
        {
            IndexType j = A.column_indices[jj];
           
            // XXX if/when MIS code filters explicit zeros we need to do that here too

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

template <typename TestMatrix, typename ExampleMatrix>
void _TestMaximalIndependentSet(const ExampleMatrix& example_matrix)
{
    typedef typename TestMatrix::memory_space MemorySpace;

    // initialize test matrix
    TestMatrix test_matrix(example_matrix);

    // allocate storage for MIS result
    cusp::array1d<int, MemorySpace> stencil(test_matrix.num_rows);

    {
        // compute MIS
        size_t num_iterations = cusp::graph::maximal_independent_set(test_matrix, stencil);

        //std::cout << "MIS(1) computed in " << num_iterations << " iterations" << std::endl;

        // check MIS for default k=1
        ASSERT_EQUAL(is_valid_mis(test_matrix, stencil), true);
    }

    {
        // compute MIS(2)
        size_t num_iterations = cusp::graph::maximal_independent_set(test_matrix, stencil, 2);

        //std::cout << "MIS(2) computed in " << num_iterations << " iterations" << std::endl;

        // check MIS(2)
        cusp::coo_matrix<int,float,MemorySpace> A(example_matrix);
        cusp::coo_matrix<int,float,MemorySpace> A2;
        cusp::multiply(A, A, A2);

        ASSERT_EQUAL(is_valid_mis(A2, stencil), true);
    }
}

void TestMaximalIndependentSetCsrDevice(void)
{
    typedef cusp::csr_matrix<int,float,cusp::device_memory> TestMatrix;
   
    // note: examples should be {0,1} matrices with 1s on the diagonal

    // two components of two nodes
    cusp::array2d<float,cusp::host_memory> A(4,4);
    A(0,0) = 1; A(0,1) = 1; A(0,2) = 0; A(0,3) = 0;
    A(1,0) = 1; A(1,1) = 1; A(1,2) = 0; A(1,3) = 0;
    A(2,0) = 0; A(2,1) = 0; A(2,2) = 1; A(2,3) = 1;
    A(3,0) = 0; A(3,1) = 0; A(3,2) = 1; A(3,3) = 1;
    
    // linear graph
    cusp::array2d<float,cusp::host_memory> B(4,4);
    B(0,0) = 1; B(0,1) = 1; B(0,2) = 0; B(0,3) = 0;
    B(1,0) = 1; B(1,1) = 1; B(1,2) = 1; B(1,3) = 0;
    B(2,0) = 0; B(2,1) = 1; B(2,2) = 1; B(2,3) = 1;
    B(3,0) = 0; B(3,1) = 0; B(3,2) = 1; B(3,3) = 1;
    
    // complete graph
    cusp::array2d<float,cusp::host_memory> C(6,6,1);

    // empty graph
    cusp::array2d<float,cusp::host_memory> D(6,6,0);

    TestMatrix E;
    cusp::gallery::poisson5pt(E, 3, 3);
    thrust::fill(E.values.begin(), E.values.end(), 1.0f);
    
    TestMatrix F;
    cusp::gallery::poisson5pt(F, 13, 17);
    thrust::fill(F.values.begin(), F.values.end(), 1.0f);

    _TestMaximalIndependentSet<TestMatrix>(A);
    _TestMaximalIndependentSet<TestMatrix>(B);
    _TestMaximalIndependentSet<TestMatrix>(C);
    _TestMaximalIndependentSet<TestMatrix>(D);
    _TestMaximalIndependentSet<TestMatrix>(E);
}
DECLARE_UNITTEST(TestMaximalIndependentSetCsrDevice);
// TODO replace with DECLARE_SPARSE_MATRIX_UNITTEST

