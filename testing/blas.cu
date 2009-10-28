#include <unittest/unittest.h>

#include <cusp/blas.h>


template <class MemorySpace>
void TestAxpy(void)
{
    cusp::array1d<float, MemorySpace> x(4);
    cusp::array1d<float, MemorySpace> y(4);

    x[0] =  7.0f;   y[0] =  0.0f; 
    x[1] =  5.0f;   y[1] = -2.0f;
    x[2] =  4.0f;   y[2] =  0.0f;
    x[3] = -3.0f;   y[3] =  5.0f;

    cusp::blas::axpy(x.begin(), x.end(), y.begin(), 2.0f);

    ASSERT_EQUAL(y[0],  14.0);
    ASSERT_EQUAL(y[1],   8.0);
    ASSERT_EQUAL(y[2],   8.0);
    ASSERT_EQUAL(y[3],  -1.0);
    
    x[0] =  7.0f;   y[0] =  0.0f; 
    x[1] =  5.0f;   y[1] = -2.0f;
    x[2] =  4.0f;   y[2] =  0.0f;
    x[3] = -3.0f;   y[3] =  5.0f;

    cusp::blas::axpy(x, y, 2.0f);
    
    ASSERT_EQUAL(y[0],  14.0);
    ASSERT_EQUAL(y[1],   8.0);
    ASSERT_EQUAL(y[2],   8.0);
    ASSERT_EQUAL(y[3],  -1.0);
}
DECLARE_HOST_DEVICE_UNITTEST(TestAxpy);


template <class MemorySpace>
void TestAxpby(void)
{
    cusp::array1d<float, MemorySpace> x(4);
    cusp::array1d<float, MemorySpace> y(4);
    cusp::array1d<float, MemorySpace> z(4,0);

    x[0] =  7.0f;   y[0] =  0.0f; 
    x[1] =  5.0f;   y[1] = -2.0f;
    x[2] =  4.0f;   y[2] =  0.0f;
    x[3] = -3.0f;   y[3] =  5.0f;

    cusp::blas::axpby(x.begin(), x.end(), y.begin(), z.begin(), 2.0f, 1.0f);

    ASSERT_EQUAL(z[0],  14.0);
    ASSERT_EQUAL(z[1],   8.0);
    ASSERT_EQUAL(z[2],   8.0);
    ASSERT_EQUAL(z[3],  -1.0);
   
    z[0] = 0.0f;
    z[1] = 0.0f;
    z[2] = 0.0f;
    z[3] = 0.0f;

    cusp::blas::axpby(x, y, z, 2.0f, 1.0f);
    
    ASSERT_EQUAL(z[0],  14.0);
    ASSERT_EQUAL(z[1],   8.0);
    ASSERT_EQUAL(z[2],   8.0);
    ASSERT_EQUAL(z[3],  -1.0);
}
DECLARE_HOST_DEVICE_UNITTEST(TestAxpby);


template <class MemorySpace>
void TestCopy(void)
{
    cusp::array1d<float, MemorySpace> x(4);
    cusp::array1d<float, MemorySpace> y(4);

    x[0] =  7.0f;   y[0] =  0.0f; 
    x[1] =  5.0f;   y[1] = -2.0f;
    x[2] =  4.0f;   y[2] =  0.0f;
    x[3] = -3.0f;   y[3] =  5.0f;

    cusp::blas::copy(x.begin(), x.end(), y.begin());

    ASSERT_EQUAL(x, y);

    x[0] =  7.0f;   y[0] =  0.0f; 
    x[1] =  5.0f;   y[1] = -2.0f;
    x[2] =  4.0f;   y[2] =  0.0f;
    x[3] = -3.0f;   y[3] =  5.0f;

    cusp::blas::copy(x, y);

    ASSERT_EQUAL(x, y);
}
DECLARE_HOST_DEVICE_UNITTEST(TestCopy);


template <class MemorySpace>
void TestDot(void)
{
    cusp::array1d<float, MemorySpace> x(6);
    cusp::array1d<float, MemorySpace> y(6);

    x[0] =  7.0f;   y[0] =  0.0f; 
    x[1] =  5.0f;   y[1] = -2.0f;
    x[2] =  4.0f;   y[2] =  0.0f;
    x[3] = -3.0f;   y[3] =  5.0f;
    x[4] =  0.0f;   y[4] =  6.0f;
    x[5] =  4.0f;   y[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::dot(x.begin(), x.end(), y.begin()), -21.0f);
    
    ASSERT_EQUAL(cusp::blas::dot(x, y), -21.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestDot);


template <class MemorySpace>
void TestFill(void)
{
    cusp::array1d<float, MemorySpace> x(4);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;

    cusp::blas::fill(x.begin(), x.end(), 1.0f);

    ASSERT_EQUAL(x[0], 1.0);
    ASSERT_EQUAL(x[1], 1.0);
    ASSERT_EQUAL(x[2], 1.0);
    ASSERT_EQUAL(x[3], 1.0);
    
    cusp::blas::fill(x, 2.0f);

    ASSERT_EQUAL(x[0], 2.0);
    ASSERT_EQUAL(x[1], 2.0);
    ASSERT_EQUAL(x[2], 2.0);
    ASSERT_EQUAL(x[3], 2.0);
}
DECLARE_HOST_DEVICE_UNITTEST(TestFill);


template <class MemorySpace>
void TestNrm2(void)
{
    cusp::array1d<float, MemorySpace> x(6);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::nrm2(x.begin(), x.end()), 10.0f);

    ASSERT_EQUAL(cusp::blas::nrm2(x), 10.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestNrm2);


template <class MemorySpace>
void TestScal(void)
{
    cusp::array1d<float, MemorySpace> x(6);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  4.0f;

    cusp::blas::scal(x.begin(), x.end(), 2.0f);

    ASSERT_EQUAL(x[0], 14.0);
    ASSERT_EQUAL(x[1], 10.0);
    ASSERT_EQUAL(x[2],  8.0);
    ASSERT_EQUAL(x[3], -6.0);
    ASSERT_EQUAL(x[4],  0.0);
    ASSERT_EQUAL(x[5],  8.0);
    
    cusp::blas::scal(x, 2.0f);

    ASSERT_EQUAL(x[0],  28.0);
    ASSERT_EQUAL(x[1],  20.0);
    ASSERT_EQUAL(x[2],  16.0);
    ASSERT_EQUAL(x[3], -12.0);
    ASSERT_EQUAL(x[4],   0.0);
    ASSERT_EQUAL(x[5],  16.0);
}
DECLARE_HOST_DEVICE_UNITTEST(TestScal);

