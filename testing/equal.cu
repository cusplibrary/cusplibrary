#include <unittest/unittest.h>
#include <cusp/equal.h>
#include <cusp/array2d.h>

void TestEqualDenseMatrix(void)
{
    cusp::array2d<float, cusp::host_memory, cusp::row_major>    A(2,2);
    cusp::array2d<float, cusp::host_memory, cusp::row_major>    B(2,3);
    cusp::array2d<float, cusp::host_memory, cusp::column_major> C(2,3);
    cusp::array2d<float, cusp::host_memory, cusp::column_major> D(2,2);

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
