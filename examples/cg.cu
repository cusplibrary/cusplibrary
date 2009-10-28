#include <cusp/csr_matrix.h>
#include <cusp/io.h>
#include <cusp/linear_operator.h>
#include <cusp/krylov/cg.h>

// where to perform the computation
typedef cusp::device MemorySpace;

int main(void)
{
    // create an empty sparse matrix structure (CSR format)
    cusp::csr_matrix<int, float, MemorySpace> A;

    // load a matrix stored in MatrixMarket format
    cusp::load_matrix_market_file(A, "../testing/data/laplacian/5pt_10x10.mtx");

    // allocate storage for solution (x) and right hand side (b)
    cusp::vector<float, MemorySpace> x(A.num_rows, 0);
    cusp::vector<float, MemorySpace> b(A.num_rows);

    // initialize right hand side
    for(int i = 0; i < A.num_rows; i++)
        b[i] = i % 2;
   
    // obtain a linear operator from matrix A and call CG
    cusp::krylov::cg(cusp::make_linear_operator(A), x, b, 1e-5f, 1000, 1);

    return 0;
}

