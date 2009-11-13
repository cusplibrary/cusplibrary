#include <cusp/csr_matrix.h>
#include <cusp/io.h>
#include <cusp/krylov/cg.h>

// where to perform the computation
typedef cusp::device_memory MemorySpace;

int main(void)
{
    // create an empty sparse matrix structure (HYB format)
    cusp::hyb_matrix<int, float, MemorySpace> A;

    // load a matrix stored in MatrixMarket format
    cusp::read_matrix_market_file(A, "5pt_10x10.mtx");

    // allocate storage for solution (x) and right hand side (b)
    cusp::array1d<float, MemorySpace> x(A.num_rows, 0);
    cusp::array1d<float, MemorySpace> b(A.num_rows, 1);

    // set stopping criteria:
    //        tolerance = 1e-6
    //  iteration_limit = 100
    cusp::default_stopping_criteria stopping_criteria(1e-6, 100);

    // set verbose flag
    bool verbose = true;

    // obtain a linear operator from matrix A and call CG
    cusp::krylov::cg(A, x, b, stopping_criteria, verbose);

    return 0;
}

