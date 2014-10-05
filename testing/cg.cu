#include <unittest/unittest.h>

#include <cusp/gallery/poisson.h>
#include <cusp/csr_matrix.h>
#include <cusp/multiply.h>
#include <cusp/krylov/cg.h>

template <class MemorySpace>
void TestConjugateGradient(void)
{
    cusp::csr_matrix<int, float, MemorySpace> A;

    cusp::gallery::poisson5pt(A, 10, 10);

    cusp::array1d<float, MemorySpace> x(A.num_rows, 0.0f);
    cusp::array1d<float, MemorySpace> b(A.num_rows, 1.0f);

    cusp::monitor<float> monitor(b, 20, 1e-4);

    cusp::krylov::cg(A, x, b, monitor);

    // check residual norm
    cusp::array1d<float, MemorySpace> residual(A.num_rows, 0.0f);
    cusp::multiply(A, x, residual);
    cusp::blas::axpby(residual, b, residual, -1.0f, 1.0f);

    ASSERT_EQUAL(cusp::blas::nrm2(residual) < 1e-4 * cusp::blas::nrm2(b), true);
}
DECLARE_HOST_DEVICE_UNITTEST(TestConjugateGradient);


template <class MemorySpace>
void TestConjugateGradientZeroResidual(void)
{
    cusp::array2d<float, MemorySpace> M(2,2);
    M(0,0) = 8;
    M(0,1) = 0;
    M(1,0) = 0;
    M(1,1) = 4;

    cusp::csr_matrix<int, float, MemorySpace> A(M);

    cusp::array1d<float, MemorySpace> x(A.num_rows, 1.0f);
    cusp::array1d<float, MemorySpace> b(A.num_rows);

    cusp::multiply(A, x, b);

    cusp::monitor<float> monitor(b, 20, 0.0f);

    cusp::krylov::cg(A, x, b, monitor);

    // check residual norm
    cusp::array1d<float, MemorySpace> residual(A.num_rows, 0.0f);
    cusp::multiply(A, x, residual);
    cusp::blas::axpby(residual, b, residual, -1.0f, 1.0f);

    ASSERT_EQUAL(monitor.converged(),        true);
    ASSERT_EQUAL(monitor.iteration_count(),     0);
    ASSERT_EQUAL(cusp::blas::nrm2(residual), 0.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestConjugateGradientZeroResidual);

