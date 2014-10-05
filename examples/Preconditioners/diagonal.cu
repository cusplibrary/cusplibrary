#include <cusp/precond/diagonal.h>
#include <cusp/krylov/cg.h>
#include <cusp/csr_matrix.h>
#include <cusp/io/matrix_market.h>

#include <iostream>

// where to perform the computation
typedef cusp::device_memory MemorySpace;

// which floating point type to use
typedef float ValueType;

int main(void)
{
    // create an empty sparse matrix structure (HYB format)
    cusp::csr_matrix<int, ValueType, MemorySpace> A;

    // load a matrix stored in MatrixMarket format
    cusp::io::read_matrix_market_file(A, "A.mtx");

    // Note: A has poorly scaled rows & columns
    // set stopping criteria (iteration_limit = 100, relative_tolerance = 1e-6, absolute_tolerance = 0, verbose = true)
    cusp::array1d<ValueType, MemorySpace> rhs(A.num_rows, 1);
    cusp::monitor<ValueType> monitor(rhs, 100, 1e-6, 0, true);

    // solve without preconditioning
    {
        std::cout << "\nSolving with no preconditioner" << std::endl;

        // allocate storage for solution (x) and right hand side (b)
        cusp::array1d<ValueType, MemorySpace> x(A.num_rows, 0);
        cusp::array1d<ValueType, MemorySpace> b(rhs);

        // solve
        cusp::krylov::cg(A, x, b, monitor);
    }

    // solve with diagonal preconditioner
    {
        std::cout << "\nSolving with diagonal preconditioner (M = D^-1)" << std::endl;

        // allocate storage for solution (x) and right hand side (b)
        cusp::array1d<ValueType, MemorySpace> x(A.num_rows, 0);
        cusp::array1d<ValueType, MemorySpace> b(rhs);

        // reset the monitor
        monitor.reset(b);

        // setup preconditioner
        cusp::precond::diagonal<ValueType, MemorySpace> M(A);

        // solve
        cusp::krylov::cg(A, x, b, monitor, M);
    }

    return 0;
}

