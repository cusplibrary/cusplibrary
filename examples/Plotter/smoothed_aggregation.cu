#include <cusp/csr_matrix.h>
#include <cusp/print.h>

#include <cusp/gallery/poisson.h>
#include <cusp/krylov/cg.h>
#include <cusp/relaxation/polynomial.h>
#include <cusp/precond/aggregation/smoothed_aggregation.h>

#include <cusp/plot/convergence_plotter.h>

int main(int argc, char ** argv)
{
    typedef int                 IndexType;
    typedef double              ValueType;
    typedef cusp::device_memory MemorySpace;

    // create an empty sparse matrix structure
    cusp::csr_matrix<IndexType, ValueType, MemorySpace> A;

    IndexType N = 256;

    // create 2D Poisson problem
    cusp::gallery::poisson5pt(A, N, N);

    // allocate storage for solution (x) and right hand side (b)
    cusp::array1d<ValueType, MemorySpace> x0(A.num_rows, 0);
    cusp::array1d<ValueType, MemorySpace> b(A.num_rows, 1);

    // set stopping criteria (iteration_limit = 10000, relative_tolerance = 1e-10)
    cusp::monitor<ValueType> monitor(b, 500, 1e-10);
    cusp::plot::convergence_plotter plotter;

    {
        // solve without preconditioner
        cusp::array1d<ValueType, MemorySpace> x(x0);
        cusp::krylov::cg(A, x, b, monitor);
    }

    plotter.append("No prec.", monitor.residuals);
    monitor.reset(b);

    // solve with smoothed aggregation algebraic multigrid preconditioner and polynomial smoother
    {
        cusp::precond::aggregation::smoothed_aggregation<IndexType, ValueType, MemorySpace> M(A);
        cusp::array1d<ValueType, MemorySpace> x(x0);
        cusp::krylov::cg(A, x, b, monitor, M);
    }

    plotter.append("With SA+jac", monitor.residuals);
    monitor.reset(b);

    // solve with smoothed aggregation algebraic multigrid preconditioner and polynomial smoother
    {
        typedef cusp::relaxation::polynomial<ValueType,MemorySpace> Smoother;
        cusp::precond::aggregation::smoothed_aggregation<IndexType, ValueType, MemorySpace, Smoother> M(A);
        cusp::array1d<ValueType, MemorySpace> x(x0);
        cusp::krylov::cg(A, x, b, monitor, M);
    }

    plotter.append("With SA+poly", monitor.residuals);
    plotter.generate("convergence_plot.html");

    return 0;
}

