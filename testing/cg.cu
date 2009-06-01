#include <unittest/unittest.h>

#include <cusp/linear_operator.h>
#include <cusp/csr_matrix.h>
#include <cusp/io.h>
#include <cusp/krylov/cg.h>

template <class MemorySpace>
void TestConjugateGradient(void)
{
    cusp::csr_matrix<int, float, MemorySpace> A;

    cusp::load_matrix_market_file(A, "data/laplacian/5pt_10x10.mtx");
    //cusp::load_matrix_market_file(csr, "data/laplacian/7pt_10x10x10.mtx");
    //cusp::load_matrix_market_file(csr, "data/laplacian/3pt_100.mtx");

    float * x = cusp::new_array<float, MemorySpace>(A.num_rows);
    float * b = cusp::new_array<float, MemorySpace>(A.num_rows);

    for(int i = 0; i < A.num_rows; i++){
        cusp::set_array_element<MemorySpace>(x, i, 0.0f);
        cusp::set_array_element<MemorySpace>(b, i, float(i % 2));
    }
        
    cusp::krylov::cg(cusp::make_linear_operator(A), x, b);

    cusp::delete_array<float, MemorySpace>(x);
    cusp::delete_array<float, MemorySpace>(b);
    cusp::deallocate_matrix(A);
}
DECLARE_HOST_DEVICE_UNITTEST(TestConjugateGradient);

