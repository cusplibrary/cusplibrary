#include <unittest/unittest.h>

#include <cusp/gallery/poisson.h>
#include <cusp/csr_matrix.h>
#include <cusp/krylov/bicgstab.h>

template <class MemorySpace>
void TestBiConjugateGradientStabilized(void)
{
    cusp::csr_matrix<int, float, MemorySpace> A;

    cusp::gallery::poisson5pt(A, 10, 10);

    cusp::array1d<float, MemorySpace> x(A.num_rows, 0.0f);
    cusp::array1d<float, MemorySpace> b(A.num_rows, 1.0f);
    
    cusp::default_stopping_criteria stopping_criteria(1e-4, 20);

    cusp::krylov::bicgstab(A, x, b, stopping_criteria);
    
    // check residual norm
    cusp::array1d<float, MemorySpace> residual(A.num_rows, 0.0f);
    cusp::spblas::spmv(A, x, residual);
    cusp::blas::axpby(residual, b, residual, -1.0f, 1.0f);

    ASSERT_EQUAL(cusp::blas::nrm2(residual) < 1e-4 * cusp::blas::nrm2(b), true);
}
DECLARE_HOST_DEVICE_UNITTEST(TestBiConjugateGradientStabilized);

