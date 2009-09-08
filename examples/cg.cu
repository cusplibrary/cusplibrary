#include <cusp/csr_matrix.h>
#include <cusp/io.h>
#include <cusp/linear_operator.h>
#include <cusp/krylov/cg.h>

// where to perform the computation
typedef cusp::device_memory MemorySpace;

int main(void)
{
    // create an empty sparse matrix structure (CSR format)
    cusp::csr_matrix<int, float, MemorySpace> A;

    // load a matrix stored in MatrixMarket format
    cusp::load_matrix_market_file(A, "../testing/data/laplacian/5pt_10x10.mtx");

    // allocate storage for solution (x) and right hand side (b)
    float * x = cusp::new_array<float, MemorySpace>(A.num_rows);
    float * b = cusp::new_array<float, MemorySpace>(A.num_rows);

    // initialize solution and right hand side
    for(int i = 0; i < A.num_rows; i++)
    {
        cusp::set_array_element<MemorySpace>(x, i, 0.0f);
        cusp::set_array_element<MemorySpace>(b, i, float(i % 2));
    }
   
    // obtain a linear operator from matrix A and call CG
    cusp::krylov::cg(cusp::make_linear_operator(A), x, b, 1e-5f, 1000, 1);

    // clean up our work space
    cusp::delete_array<float, MemorySpace>(x);
    cusp::delete_array<float, MemorySpace>(b);
    cusp::deallocate_matrix(A);

    return 0;
}

