#include <unittest/unittest.h>

#include <cusp/multiply.h>
#include <cusp/coo_matrix.h>
#include <cusp/print.h>


template <typename SparseMatrixType, typename DenseMatrixType>
void CompareResults(SparseMatrixType test, DenseMatrixType A, DenseMatrixType B)
{
    DenseMatrixType C;
    cusp::multiply(A, B, C);

    SparseMatrixType _A(A), _B(B), _C;
    cusp::multiply(_A, _B, _C);

    ASSERT_EQUAL(C == DenseMatrixType(_C), true);
}

template <class Space>
void TestMatrixMultiply(void)
{
    cusp::array2d<float,cusp::host_memory> A(3,2);
    A(0,0) = 1.0; A(0,1) = 2.0;
    A(1,0) = 3.0; A(1,1) = 0.0;
    A(2,0) = 5.0; A(2,1) = 6.0;
    
    cusp::array2d<float,cusp::host_memory> B(2,4);
    B(0,0) = 0.0; B(0,1) = 2.0; B(0,2) = 3.0; B(0,3) = 4.0;
    B(1,0) = 5.0; B(1,1) = 0.0; B(1,2) = 0.0; B(1,3) = 8.0;

    cusp::array2d<float,cusp::host_memory> C(2,2);
    C(0,0) = 0.0; C(0,1) = 0.0;
    C(1,0) = 3.0; C(1,1) = 5.0;
    
    cusp::array2d<float,cusp::host_memory> D(2,1);
    D(0,0) = 2.0;
    D(1,0) = 3.0;
    
    cusp::array2d<float,cusp::host_memory> E(2,2);
    E(0,0) = 0.0; E(0,1) = 0.0;
    E(1,0) = 0.0; E(1,1) = 0.0;
    
    cusp::array2d<float,cusp::host_memory> F(2,3);
    F(0,0) = 0.0; F(0,1) = 1.5; F(0,2) = 3.0;
    F(1,0) = 0.5; F(1,1) = 0.0; F(1,2) = 0.0;
   
    CompareResults(cusp::coo_matrix<int, float, Space>(), A, B);
    CompareResults(cusp::coo_matrix<int, float, Space>(), A, C);
    CompareResults(cusp::coo_matrix<int, float, Space>(), A, D);
    CompareResults(cusp::coo_matrix<int, float, Space>(), A, E);
    CompareResults(cusp::coo_matrix<int, float, Space>(), A, F);

    CompareResults(cusp::coo_matrix<int, float, Space>(), C, C);
    CompareResults(cusp::coo_matrix<int, float, Space>(), C, D);
    CompareResults(cusp::coo_matrix<int, float, Space>(), C, E);
    CompareResults(cusp::coo_matrix<int, float, Space>(), C, F);

    CompareResults(cusp::coo_matrix<int, float, Space>(), E, B);
    CompareResults(cusp::coo_matrix<int, float, Space>(), E, C);
    CompareResults(cusp::coo_matrix<int, float, Space>(), E, D);
    CompareResults(cusp::coo_matrix<int, float, Space>(), E, E);
    CompareResults(cusp::coo_matrix<int, float, Space>(), E, F);

    CompareResults(cusp::coo_matrix<int, float, Space>(), F, A);
}
DECLARE_HOST_DEVICE_UNITTEST(TestMatrixMultiply);

