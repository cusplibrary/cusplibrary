#include <unittest/unittest.h>

#include <cusp/gallery/poisson.h>

#include <cusp/print.h>

void TestPoisson5pt(void)
{
    // grid is 2x3
    // [45]
    // [23]
    // [01]

    cusp::dia_matrix<int, float, cusp::host_memory> matrix;
    cusp::gallery::poisson5pt(matrix, 2, 3);

    // convert result to array2d
    cusp::array2d<float, cusp::host_memory> R(matrix);
    cusp::array2d<float, cusp::host_memory> E(6,6);

    E(0,0) =  4;
    E(0,1) = -1;
    E(0,2) = -1;
    E(0,3) =  0;
    E(0,4) =  0;
    E(0,5) =  0;
    E(1,0) = -1;
    E(1,1) =  4;
    E(1,2) =  0;
    E(1,3) = -1;
    E(1,4) =  0;
    E(1,5) =  0;
    E(2,0) = -1;
    E(2,1) =  0;
    E(2,2) =  4;
    E(2,3) = -1;
    E(2,4) = -1;
    E(2,5) =  0;
    E(3,0) =  0;
    E(3,1) = -1;
    E(3,2) = -1;
    E(3,3) =  4;
    E(3,4) =  0;
    E(3,5) = -1;
    E(4,0) =  0;
    E(4,1) =  0;
    E(4,2) = -1;
    E(4,3) =  0;
    E(4,4) =  4;
    E(4,5) = -1;
    E(5,0) =  0;
    E(5,1) =  0;
    E(5,2) =  0;
    E(5,3) = -1;
    E(5,4) = -1;
    E(5,5) =  4;

    ASSERT_EQUAL_QUIET(R, E);
}
DECLARE_UNITTEST(TestPoisson5pt);

