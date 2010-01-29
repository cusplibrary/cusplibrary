#include <unittest/unittest.h>

#include <cusp/multiply.h>
#include <cusp/linear_operator.h>
#include <cusp/coo_matrix.h>
#include <cusp/print.h>
#include <cusp/gallery/poisson.h>

/////////////////////////////////////////
// Sparse Matrix-Matrix Multiplication //
/////////////////////////////////////////

template <typename SparseMatrixType, typename DenseMatrixType>
void CompareSparseMatrixMatrixMultiply(SparseMatrixType test, DenseMatrixType A, DenseMatrixType B)
{
    DenseMatrixType C;
    cusp::multiply(A, B, C);

    SparseMatrixType _A(A), _B(B), _C;
    cusp::multiply(_A, _B, _C);

    ASSERT_EQUAL(C == DenseMatrixType(_C), true);
}

template <typename TestMatrix>
void TestSparseMatrixMatrixMultiply(void)
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
    
    cusp::array2d<float,cusp::host_memory> G;
    cusp::gallery::poisson5pt(G, 4, 6);

    cusp::array2d<float,cusp::host_memory> H;
    cusp::gallery::poisson5pt(H, 8, 3);
   
    CompareSparseMatrixMatrixMultiply(TestMatrix(), A, B);
    CompareSparseMatrixMatrixMultiply(TestMatrix(), A, C);
    CompareSparseMatrixMatrixMultiply(TestMatrix(), A, D);
    CompareSparseMatrixMatrixMultiply(TestMatrix(), A, E);
    CompareSparseMatrixMatrixMultiply(TestMatrix(), A, F);
    CompareSparseMatrixMatrixMultiply(TestMatrix(), C, C);
    CompareSparseMatrixMatrixMultiply(TestMatrix(), C, D);
    CompareSparseMatrixMatrixMultiply(TestMatrix(), C, E);
    CompareSparseMatrixMatrixMultiply(TestMatrix(), C, F);
    CompareSparseMatrixMatrixMultiply(TestMatrix(), E, B);
    CompareSparseMatrixMatrixMultiply(TestMatrix(), E, C);
    CompareSparseMatrixMatrixMultiply(TestMatrix(), E, D);
    CompareSparseMatrixMatrixMultiply(TestMatrix(), E, E);
    CompareSparseMatrixMatrixMultiply(TestMatrix(), E, F);
    CompareSparseMatrixMatrixMultiply(TestMatrix(), F, A);
    CompareSparseMatrixMatrixMultiply(TestMatrix(), G, G);
    CompareSparseMatrixMatrixMultiply(TestMatrix(), G, H);
    CompareSparseMatrixMatrixMultiply(TestMatrix(), H, H);
}

template <typename Space>
void TestSparseMatrixMatrixMultiplyCoo(void)
{
    TestSparseMatrixMatrixMultiply< cusp::coo_matrix<int,float,Space> >();
}
DECLARE_HOST_DEVICE_UNITTEST(TestSparseMatrixMatrixMultiplyCoo);



/////////////////////////////////////////
// Sparse Matrix-Vector Multiplication //
/////////////////////////////////////////

template <class TestMatrix>
void TestSparseMatrixVectorMultiply()
{
    typedef typename TestMatrix::memory_space MemorySpace;

    // initialize example matrix
    cusp::array2d<float, cusp::host_memory> A(5,4);
    A(0,0) = 13; A(0,1) = 80; A(0,2) =  0; A(0,3) =  0; 
    A(1,0) =  0; A(1,1) = 27; A(1,2) =  0; A(1,3) =  0;
    A(2,0) = 55; A(2,1) =  0; A(2,2) = 24; A(2,3) = 42;
    A(3,0) =  0; A(3,1) = 69; A(3,2) =  0; A(3,3) = 83;
    A(4,0) =  0; A(4,1) =  0; A(4,2) = 27; A(4,3) =  0;

    // convert to desired format
    TestMatrix test_matrix = A;

    // allocate vectors
    cusp::array1d<float, MemorySpace> x(4);
    cusp::array1d<float, MemorySpace> y(5);
   
    // initialize input and output vectors
    x[0] = 1.0f; y[0] = 10.0f; 
    x[1] = 2.0f; y[1] = 20.0f;
    x[2] = 3.0f; y[2] = 30.0f;
    x[3] = 4.0f; y[3] = 40.0f;
                 y[4] = 50.0f;

    cusp::multiply(test_matrix, x, y);

    ASSERT_EQUAL(y[0], 173.0f);
    ASSERT_EQUAL(y[1],  54.0f);
    ASSERT_EQUAL(y[2], 295.0f);
    ASSERT_EQUAL(y[3], 470.0f);
    ASSERT_EQUAL(y[4],  81.0f);
}
DECLARE_SPARSE_MATRIX_UNITTEST(TestSparseMatrixVectorMultiply);


template <class MemorySpace>
void TestMultiplyIdentityOperator(void)
{
    cusp::array1d<float, MemorySpace> x(4);
    cusp::array1d<float, MemorySpace> y(4);

    x[0] =  7.0f;   y[0] =  0.0f; 
    x[1] =  5.0f;   y[1] = -2.0f;
    x[2] =  4.0f;   y[2] =  0.0f;
    x[3] = -3.0f;   y[3] =  5.0f;

    cusp::identity_operator<float, MemorySpace> A(4,4);
    
    cusp::multiply(A, x, y);

    ASSERT_EQUAL(y[0],  7.0f);
    ASSERT_EQUAL(y[1],  5.0f);
    ASSERT_EQUAL(y[2],  4.0f);
    ASSERT_EQUAL(y[3], -3.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestMultiplyIdentityOperator);

