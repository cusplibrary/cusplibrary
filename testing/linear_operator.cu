#include <unittest/unittest.h>

#include <cusp/linear_operator.h>

template <class MemorySpace>
void TestLinearOperator(void)
{
    cusp::array1d<float, MemorySpace> x(4);
    cusp::array1d<float, MemorySpace> y(4);
    cusp::array1d<float, MemorySpace> z(4);

    x[0] =  7.0f;   y[0] =  0.0f; 
    x[1] =  5.0f;   y[1] = -2.0f;
    x[2] =  4.0f;   y[2] =  0.0f;
    x[3] = -3.0f;   y[3] =  5.0f;

    cusp::linear_operator<float, MemorySpace> A(4,4);

    // linear_operator throws exceptions
    ASSERT_THROWS(A.multiply(x, y),                cusp::not_implemented_exception);
    ASSERT_THROWS(A.multiply(x, y, z, 1.0f, 2.0f), cusp::not_implemented_exception);
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

    A.multiply(x, z);

    ASSERT_EQUAL(z[0],  7.0f);
    ASSERT_EQUAL(z[1],  5.0f);
    ASSERT_EQUAL(z[2],  4.0f);
    ASSERT_EQUAL(z[3], -3.0f);

    A.multiply(x, y, z, 1.0f, 2.0f);
    
    ASSERT_EQUAL(z[0],  7.0f);
    ASSERT_EQUAL(z[1],  1.0f);
    ASSERT_EQUAL(z[2],  4.0f);
    ASSERT_EQUAL(z[3],  7.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestIdentityOperator);

