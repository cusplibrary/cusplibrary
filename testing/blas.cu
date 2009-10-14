#include <unittest/unittest.h>

#include <cusp/blas.h>


template <class MemorySpace>
void TestAxpy(void)
{
    cusp::vector<float, MemorySpace> x(6);
    cusp::vector<float, MemorySpace> y(6);

    x[0] =  7.0f;   y[0] =  0.0f; 
    x[1] =  5.0f;   y[1] = -2.0f;
    x[2] =  4.0f;   y[2] =  0.0f;
    x[3] = -3.0f;   y[3] =  5.0f;
    x[4] =  0.0f;   y[4] =  6.0f;
    x[5] =  4.0f;   y[5] =  1.0f;

    cusp::blas<MemorySpace>::axpy(6, 2.0f, thrust::raw_pointer_cast(&x[0]), 
                                           thrust::raw_pointer_cast(&y[0]));

    ASSERT_EQUAL(y[0],  14.0);
    ASSERT_EQUAL(y[1],   8.0);
    ASSERT_EQUAL(y[2],   8.0);
    ASSERT_EQUAL(y[3],  -1.0);
    ASSERT_EQUAL(y[4],   6.0);
    ASSERT_EQUAL(y[5],   9.0);
}
DECLARE_HOST_DEVICE_UNITTEST(TestAxpy);

template <class MemorySpace>
void TestCopy(void)
{
    cusp::vector<float, MemorySpace> x(6);
    cusp::vector<float, MemorySpace> y(6);

    x[0] =  7.0f;   y[0] =  0.0f; 
    x[1] =  5.0f;   y[1] = -2.0f;
    x[2] =  4.0f;   y[2] =  0.0f;
    x[3] = -3.0f;   y[3] =  5.0f;
    x[4] =  0.0f;   y[4] =  6.0f;
    x[5] =  4.0f;   y[5] =  1.0f;

    cusp::blas<MemorySpace>::copy(6, thrust::raw_pointer_cast(&x[0]), 
                                     thrust::raw_pointer_cast(&y[0]));

    y[0] =  7.0f;
    y[1] =  5.0f;
    y[2] =  4.0f;
    y[3] = -3.0f;
    y[4] =  0.0f;
    y[5] =  4.0f;
}
DECLARE_HOST_DEVICE_UNITTEST(TestCopy);


template <class MemorySpace>
void TestDot(void)
{
    cusp::vector<float, MemorySpace> x(6);
    cusp::vector<float, MemorySpace> y(6);

    x[0] =  7.0f;   y[0] =  0.0f; 
    x[1] =  5.0f;   y[1] = -2.0f;
    x[2] =  4.0f;   y[2] =  0.0f;
    x[3] = -3.0f;   y[3] =  5.0f;
    x[4] =  0.0f;   y[4] =  6.0f;
    x[5] =  4.0f;   y[5] =  1.0f;

    float result = cusp::blas<MemorySpace>::dot(6, thrust::raw_pointer_cast(&x[0]), 
                                                   thrust::raw_pointer_cast(&y[0]));

    ASSERT_EQUAL(result, -21.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestDot);


template <class MemorySpace>
void TestFill(void)
{
    cusp::vector<float, MemorySpace> x(6);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  4.0f;

    cusp::blas<MemorySpace>::fill(6, 1.0f, thrust::raw_pointer_cast(&x[0]));

    ASSERT_EQUAL(x[0], 1.0);
    ASSERT_EQUAL(x[1], 1.0);
    ASSERT_EQUAL(x[2], 1.0);
    ASSERT_EQUAL(x[3], 1.0);
    ASSERT_EQUAL(x[4], 1.0);
    ASSERT_EQUAL(x[5], 1.0);
}
DECLARE_HOST_DEVICE_UNITTEST(TestFill);


template <class MemorySpace>
void TestNrm2(void)
{
    cusp::vector<float, MemorySpace> x(6);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  1.0f;

    float result = cusp::blas<MemorySpace>::nrm2(6, thrust::raw_pointer_cast(&x[0]));

    ASSERT_EQUAL(result, 10.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestNrm2);


template <class MemorySpace>
void TestScal(void)
{
    cusp::vector<float, MemorySpace> x(6);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  4.0f;

    cusp::blas<MemorySpace>::scal(6, 2.0f, thrust::raw_pointer_cast(&x[0]));

    ASSERT_EQUAL(x[0], 14.0);
    ASSERT_EQUAL(x[1], 10.0);
    ASSERT_EQUAL(x[2],  8.0);
    ASSERT_EQUAL(x[3], -6.0);
    ASSERT_EQUAL(x[4],  0.0);
    ASSERT_EQUAL(x[5],  8.0);
}
DECLARE_HOST_DEVICE_UNITTEST(TestScal);

