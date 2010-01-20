#include <unittest/unittest.h>

#include <cusp/graph/maximal_independent_set.h>
#include <cusp/gallery/poisson.h> 

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

void TestCsrMaximalIndependentSet(void)
{
    typedef cusp::csr_matrix<int,float,cusp::device_memory> TestMatrix;
    typedef typename TestMatrix::memory_space MemorySpace;

    // initialize test matrix
    TestMatrix A;
    cusp::gallery::poisson5pt(A, 13, 17);

    // allocate storage for MIS result
    cusp::array1d<int, MemorySpace> stencil(A.num_rows, 0);

    // compute MIS
    size_t num_iterations = cusp::graph::maximal_independent_set(A, stencil);
    
    //std::cout << "Luby's algorithm finished in " << num_iterations << " iterations" << std::endl;

    // check MIS
    ASSERT_EQUAL(is_valid_mis(A, stencil), true);
}
DECLARE_UNITTEST(TestCsrMaximalIndependentSet);

