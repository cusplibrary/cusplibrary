#include <cusp/csr_matrix.h>
#include <cusp/monitor.h>
#include <cusp/krylov/cg.h>
#include <cusp/gallery/poisson.h>

int main(void)
{
    // create an empty sparse matrix structure (CSR format)
    cusp::csr_matrix<int, float, cusp::device_memory> A;

    // initialize matrix
    cusp::gallery::poisson5pt(A, 10, 10);

    // allocate storage for solution (x) and right hand side (b)
    cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0);
    cusp::array1d<float, cusp::device_memory> b(A.num_rows, 1);

    // set stopping criteria:
    //  iteration_limit    = 100
    //  relative_tolerance = 1e-6
    cusp::default_monitor<float> monitor(b, 100, 1e-6);

    // solve the linear system A x = b
    cusp::krylov::cg(A, x, b, monitor);

    // report solver results
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

    return 0;
}

