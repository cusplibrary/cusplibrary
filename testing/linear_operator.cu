#include <unittest/unittest.h>

#include <cusp/linear_operator.h>

template <class MemorySpace>
void TestLinearOperator(void)
{
    cusp::array1d<float, MemorySpace> x(4);
    cusp::array1d<float, MemorySpace> y(4);

    cusp::linear_operator<float, MemorySpace> A(4,3);

    ASSERT_EQUAL(A.num_rows, 4);
    ASSERT_EQUAL(A.num_cols, 3);

    // linear_operator throws exceptions
    ASSERT_THROWS(A(x, y), cusp::not_implemented_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestLinearOperator);

template <class MemorySpace>
void TestIdentityOperator(void)
{
    cusp::array1d<float, MemorySpace> x(4);
    cusp::array1d<float, MemorySpace> y(4);
    cusp::array1d<float, MemorySpace> z(4);

    x[0] =  7.0f;   y[0] =  0.0f; 
    x[1] =  5.0f;   y[1] = -2.0f;
    x[2] =  4.0f;   y[2] =  0.0f;
    x[3] = -3.0f;   y[3] =  5.0f;

    cusp::identity_operator<float, MemorySpace> A(4,4);

    A(x, y);

    ASSERT_EQUAL(y[0],  7.0f);
    ASSERT_EQUAL(y[1],  5.0f);
    ASSERT_EQUAL(y[2],  4.0f);
    ASSERT_EQUAL(y[3], -3.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestIdentityOperator);

