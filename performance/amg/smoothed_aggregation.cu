#include <cusp/precond/smoothed_aggregation.h>
#include <cusp/krylov/cg.h>
#include <cusp/gallery/poisson.h>
#include <cusp/csr_matrix.h>

#include <iostream>

#include "../timer.h"

template <typename Monitor>
void report_status(Monitor& monitor)
{
    if (monitor.converged())
    {
        std::cout << "Solver converged to " << monitor.tolerance() << " tolerance";
        std::cout << " after " << monitor.iteration_count() << " iterations";
        std::cout << " (" << monitor.residual_norm() << " final residual)" << std::endl;
    }
    else
    {
        std::cout << "Solver reached iteration limit " << monitor.iteration_limit() << " before converging";
        std::cout << " to " << monitor.tolerance() << " tolerance ";
        std::cout << " (" << monitor.residual_norm() << " final residual)" << std::endl;
    }
}

int main(int argc, char ** argv)
{
    typedef int                 IndexType;
    typedef double              ValueType;
    typedef cusp::device_memory MemorySpace;

    // create an empty sparse matrix structure
    cusp::coo_matrix<IndexType, ValueType, MemorySpace> A;

    IndexType N = 1024;

    // create 2D Poisson problem
    cusp::gallery::poisson5pt(A, N, N);

    // solve without preconditioning
    {
        std::cout << "\nSolving with no preconditioner" << std::endl;
    
        // allocate storage for solution (x) and right hand side (b)
        cusp::array1d<ValueType, MemorySpace> x(A.num_rows, 0);
        cusp::array1d<ValueType, MemorySpace> b(A.num_rows, 1);

        // set stopping criteria (iteration_limit = 10000, relative_tolerance = 1e-10)
        cusp::default_monitor<ValueType> monitor(b, 10000, 1e-10);
        
        // solve
        timer t0;
        cusp::krylov::cg(A, x, b, monitor);
        std::cout << "solved system  in " << t0.milliseconds_elapsed() << " ms " << std::endl;

        // report status
        report_status(monitor);
    }

    // solve with smoothed aggregation algebraic multigrid preconditioner
    {
        std::cout << "\nSolving with smoothed aggregation preconditioner" << std::endl;
        
        // allocate storage for solution (x) and right hand side (b)
        cusp::array1d<ValueType, MemorySpace> x(A.num_rows, 0);
        cusp::array1d<ValueType, MemorySpace> b(A.num_rows, 1);

        // set stopping criteria (iteration_limit = 1000, relative_tolerance = 1e-10)
        cusp::default_monitor<ValueType> monitor(b, 1000, 1e-10);

        // setup preconditioner
        timer t0;
        cusp::precond::smoothed_aggregation<IndexType, ValueType, MemorySpace> M(A);
        std::cout << "constructed hierarchy in " << t0.milliseconds_elapsed() << " ms " << std::endl;

        // solve
        timer t1;
        cusp::krylov::cg(A, x, b, monitor, M);
        std::cout << "solved system  in " << t1.milliseconds_elapsed() << " ms " << std::endl;
        
        // report status
        report_status(monitor);
    }

    return 0;
}

