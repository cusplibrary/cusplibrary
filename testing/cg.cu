#include <unittest/unittest.h>

#include <cusp/gallery/poisson.h>
#include <cusp/linear_operator.h>
#include <cusp/csr_matrix.h>
#include <cusp/krylov/cg.h>

template <class MemorySpace>
void TestConjugateGradient(void)
{
    cusp::csr_matrix<int, float, MemorySpace> A;

    cusp::gallery::poisson5pt(A, 50, 50);

    cusp::array1d<float, MemorySpace> x(A.num_rows, 0.0f);
    cusp::array1d<float, MemorySpace> b(A.num_rows, 1.0f);

    cusp::krylov::cg(cusp::make_linear_operator(A), x, b, 1e-5, 100);
}
DECLARE_HOST_DEVICE_UNITTEST(TestConjugateGradient);

