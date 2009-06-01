#include <unittest/unittest.h>

#include <cusp/io.h>
#include <cusp/csr_matrix.h>
#include <cusp/arithmetic.h>


template <class MemorySpace>
void TestMatrixAddition(void)
{
    //cusp::csr_matrix<int, float, MemorySpace> A, B, C;

    //cusp::load_matrix_market_file(A, "data/laplacian/5pt_10x10.mtx");
    //cusp::load_matrix_market_file(B, "data/laplacian/5pt_10x10.mtx");

    //cusp::add(C, A, B);

}
DECLARE_HOST_DEVICE_UNITTEST(TestMatrixAddition);
