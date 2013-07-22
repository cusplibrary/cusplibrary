#include <cusp/relaxation/jacobi.h>
#include <cusp/relaxation/polynomial.h>
#include <cusp/precond/aggregation/smoothed_aggregation.h>
#include <cusp/krylov/cg.h>
#include <cusp/gallery/poisson.h>
#include <cusp/csr_matrix.h>

#include <iostream>

#include "../timer.h"

template<typename IndexType, typename ValueType, typename MemorySpace>
class unsmoothed_aggregation_options
    : public cusp::precond::aggregation::smoothed_aggregation_options<IndexType,ValueType,MemorySpace>
{
    typedef cusp::precond::aggregation::smoothed_aggregation_options<IndexType,ValueType,MemorySpace> Parent;
    typedef typename Parent::MatrixType MatrixType;

public:

    unsmoothed_aggregation_options() : Parent() {}

    virtual void smooth_prolongator(const MatrixType& A, const MatrixType& T, MatrixType& P, ValueType& rho_DinvA) const
    {
        P = T;
    }
};

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

template<typename MatrixType, typename Prec>
void run_amg(const MatrixType& A, Prec& M)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    // allocate storage for solution (x) and right hand side (b)
    cusp::array1d<ValueType, MemorySpace> x(A.num_rows, 0);
    cusp::array1d<ValueType, MemorySpace> b(A.num_rows, 1);

    // set stopping criteria (iteration_limit = 1000, relative_tolerance = 1e-10)
    cusp::default_monitor<ValueType> monitor(b, 1000, 1e-10);

    // solve
    timer t1;
    cusp::krylov::cg(A, x, b, monitor, M);
    std::cout << "solved system  in " << t1.milliseconds_elapsed() << " ms " << std::endl;

    // report status
    report_status(monitor);
}

int main(int argc, char ** argv)
{
    typedef int                 IndexType;
    typedef float               ValueType;
    typedef cusp::device_memory MemorySpace;

    // create an empty sparse matrix structure
    cusp::hyb_matrix<IndexType, ValueType, MemorySpace> A;

    IndexType N = 256;

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
        std::cout << "\nSolving with smoothed aggregation preconditioner and jacobi smoother" << std::endl;

        // setup preconditioner
        timer t0;
        cusp::precond::aggregation::smoothed_aggregation<IndexType, ValueType, MemorySpace> M(A);
        std::cout << "constructed hierarchy in " << t0.milliseconds_elapsed() << " ms " << std::endl;

        run_amg(A,M);
    }

    // solve with smoothed aggregation algebraic multigrid preconditioner and polynomial smoother
    {
        typedef cusp::relaxation::polynomial<ValueType,MemorySpace> Smoother;
        std::cout << "\nSolving with smoothed aggregation preconditioner and polynomial smoother" << std::endl;

        timer t0;
        cusp::precond::aggregation::smoothed_aggregation<IndexType, ValueType, MemorySpace, Smoother> M(A);
        std::cout << "constructed hierarchy in " << t0.milliseconds_elapsed() << " ms " << std::endl;

        run_amg(A,M);
    }

    // solve with unsmoothed aggregation algebraic multigrid preconditioner and polynomial smoother
    {
        std::cout << "\nSolving with unsmoothed aggregation preconditioner" << std::endl;

        unsmoothed_aggregation_options<IndexType,ValueType,MemorySpace> opts;

        timer t0;
        cusp::precond::aggregation::smoothed_aggregation<IndexType, ValueType, MemorySpace> M(A,opts);
        std::cout << "constructed hierarchy in " << t0.milliseconds_elapsed() << " ms " << std::endl;

        run_amg(A,M);
    }

    return 0;
}

