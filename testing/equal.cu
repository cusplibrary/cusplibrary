#include <unittest/unittest.h>
#include <cusp/equal.h>

void TestEqualDenseMatrix(void)
{
    cusp::dense_matrix<float, cusp::host_memory, cusp::row_major>    A;
    cusp::dense_matrix<float, cusp::host_memory, cusp::row_major>    B;
    cusp::dense_matrix<float, cusp::host_memory, cusp::column_major> C;
    cusp::dense_matrix<float, cusp::host_memory, cusp::column_major> D;

    cusp::allocate_matrix(A, 2, 3);
    cusp::allocate_matrix(B, 2, 3);
    cusp::allocate_matrix(C, 2, 3);
    cusp::allocate_matrix(D, 2, 2);

    A(0,0) = 1;  A(0,1) = 2;  A(0,2) = 3;
    A(1,0) = 4;  A(1,1) = 5;  A(1,2) = 6;
    
    B(0,0) = 1;  B(0,1) = 2;  B(0,2) = 3;
    B(1,0) = 7;  B(1,1) = 5;  B(1,2) = 6;
    
    C(0,0) = 1;  C(0,1) = 2;  C(0,2) = 3;
    C(1,0) = 4;  C(1,1) = 5;  C(1,2) = 6;
    
    D(0,0) = 1;  D(0,1) = 2;
    D(1,0) = 4;  D(1,1) = 5;

    ASSERT_EQUAL(cusp::equal(A,A),  true);
    ASSERT_EQUAL(cusp::equal(B,B),  true);
    ASSERT_EQUAL(cusp::equal(C,C),  true);
    ASSERT_EQUAL(cusp::equal(D,D),  true);

    ASSERT_EQUAL(cusp::equal(A,B), false);
    ASSERT_EQUAL(cusp::equal(A,C),  true);
    ASSERT_EQUAL(cusp::equal(A,D), false);
    ASSERT_EQUAL(cusp::equal(B,C), false);
    ASSERT_EQUAL(cusp::equal(B,D), false);
    ASSERT_EQUAL(cusp::equal(C,D), false);
}
