#include <unittest/unittest.h>

#include <cusp/complex.h>
#include <cusp/blas/blas.h>

template <class MemorySpace>
void TestAxpy(void)
{
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    Array x(4);
    Array y(4);


    x[0] =  7.0f;
    y[0] =  0.0f;
    x[1] =  5.0f;
    y[1] = -2.0f;
    x[2] =  4.0f;
    y[2] =  0.0f;
    x[3] = -3.0f;
    y[3] =  5.0f;

    cusp::blas::axpy(x, y, 2.0f);

    ASSERT_EQUAL(y[0],  14.0);
    ASSERT_EQUAL(y[1],   8.0);
    ASSERT_EQUAL(y[2],   8.0);
    ASSERT_EQUAL(y[3],  -1.0);

    View view_x(x);
    View view_y(y);

    cusp::blas::axpy(view_x, view_y, 2.0f);

    ASSERT_EQUAL(y[0],  28.0);
    ASSERT_EQUAL(y[1],  18.0);
    ASSERT_EQUAL(y[2],  16.0);
    ASSERT_EQUAL(y[3],  -7.0);

    // test size checking
    Array w(3);
    ASSERT_THROWS(cusp::blas::axpy(x, w, 1.0f), cusp::invalid_input_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestAxpy);


template <class MemorySpace>
void TestAxpby(void)
{
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    Array x(4);
    Array y(4);
    Array z(4,0);

    x[0] =  7.0f;
    y[0] =  0.0f;
    x[1] =  5.0f;
    y[1] = -2.0f;
    x[2] =  4.0f;
    y[2] =  0.0f;
    x[3] = -3.0f;
    y[3] =  5.0f;


    cusp::blas::axpby(x, y, z, 2.0f, 1.0f);

    ASSERT_EQUAL(z[0],  14.0);
    ASSERT_EQUAL(z[1],   8.0);
    ASSERT_EQUAL(z[2],   8.0);
    ASSERT_EQUAL(z[3],  -1.0);

    z[0] = 0.0f;
    z[1] = 0.0f;
    z[2] = 0.0f;
    z[3] = 0.0f;

    View view_x(x);
    View view_y(y);
    View view_z(z);

    cusp::blas::axpby(view_x, view_y, view_z, 2.0f, 1.0f);

    ASSERT_EQUAL(z[0],  14.0);
    ASSERT_EQUAL(z[1],   8.0);
    ASSERT_EQUAL(z[2],   8.0);
    ASSERT_EQUAL(z[3],  -1.0);

    // test size checking
    Array w(3);
    ASSERT_THROWS(cusp::blas::axpby(x, y, w, 2.0f, 1.0f), cusp::invalid_input_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestAxpby);


template <class MemorySpace>
void TestAxpbypcz(void)
{
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    Array x(4);
    Array y(4);
    Array z(4);
    Array w(4,0);

    x[0] =  7.0f;
    y[0] =  0.0f;
    z[0] =  1.0f;
    x[1] =  5.0f;
    y[1] = -2.0f;
    z[1] =  0.0f;
    x[2] =  4.0f;
    y[2] =  0.0f;
    z[2] =  3.0f;
    x[3] = -3.0f;
    y[3] =  5.0f;
    z[3] = -2.0f;


    cusp::blas::axpbypcz(x, y, z, w, 2.0f, 1.0f, 3.0f);

    ASSERT_EQUAL(w[0],  17.0);
    ASSERT_EQUAL(w[1],   8.0);
    ASSERT_EQUAL(w[2],  17.0);
    ASSERT_EQUAL(w[3],  -7.0);

    w[0] = 0.0f;
    w[1] = 0.0f;
    w[2] = 0.0f;
    w[3] = 0.0f;

    View view_x(x);
    View view_y(y);
    View view_z(z);
    View view_w(w);

    cusp::blas::axpbypcz(view_x, view_y, view_z, view_w, 2.0f, 1.0f, 3.0f);

    ASSERT_EQUAL(w[0],  17.0);
    ASSERT_EQUAL(w[1],   8.0);
    ASSERT_EQUAL(w[2],  17.0);
    ASSERT_EQUAL(w[3],  -7.0);

    // test size checking
    Array output(3);
    ASSERT_THROWS(cusp::blas::axpbypcz(x, y, z, output, 2.0f, 1.0f, 3.0f), cusp::invalid_input_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestAxpbypcz);


template <class MemorySpace>
void TestXmy(void)
{
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    Array x(4);
    Array y(4);
    Array z(4,0);

    x[0] =  7.0f;
    y[0] =  0.0f;
    x[1] =  5.0f;
    y[1] = -2.0f;
    x[2] =  4.0f;
    y[2] =  0.0f;
    x[3] = -3.0f;
    y[3] =  5.0f;


    cusp::blas::xmy(x, y, z);

    ASSERT_EQUAL(z[0],   0.0f);
    ASSERT_EQUAL(z[1], -10.0f);
    ASSERT_EQUAL(z[2],   0.0f);
    ASSERT_EQUAL(z[3], -15.0f);

    z[0] = 0.0f;
    z[1] = 0.0f;
    z[2] = 0.0f;
    z[3] = 0.0f;

    View view_x(x);
    View view_y(y);
    View view_z(z);

    cusp::blas::xmy(view_x, view_y, view_z);

    ASSERT_EQUAL(z[0],   0.0f);
    ASSERT_EQUAL(z[1], -10.0f);
    ASSERT_EQUAL(z[2],   0.0f);
    ASSERT_EQUAL(z[3], -15.0f);

    // test size checking
    Array output(3);
    ASSERT_THROWS(cusp::blas::xmy(x, y, output), cusp::invalid_input_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestXmy);


template <class MemorySpace>
void TestCopy(void)
{
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    Array x(4);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;

    {
        Array y(4, -1);
        cusp::blas::copy(x, y);
        ASSERT_EQUAL(x, y);
    }

    {
        Array y(4, -1);
        cusp::blas::copy(View(x), View(y));
        ASSERT_EQUAL(x, y);
    }

    // test size checking
    cusp::array1d<float, MemorySpace> w(3);
    ASSERT_THROWS(cusp::blas::copy(w, x), cusp::invalid_input_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestCopy);


template <class MemorySpace>
void TestDot(void)
{
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    Array x(6);
    Array y(6);

    x[0] =  7.0f;
    y[0] =  0.0f;
    x[1] =  5.0f;
    y[1] = -2.0f;
    x[2] =  4.0f;
    y[2] =  0.0f;
    x[3] = -3.0f;
    y[3] =  5.0f;
    x[4] =  0.0f;
    y[4] =  6.0f;
    x[5] =  4.0f;
    y[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::dot(x, y), -21.0f);

    ASSERT_EQUAL(cusp::blas::dot(View(x), View(y)), -21.0f);

    // test size checking
    cusp::array1d<float, MemorySpace> w(3);
    ASSERT_THROWS(cusp::blas::dot(x, w), cusp::invalid_input_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestDot);


template <class MemorySpace>
void TestDotc(void)
{
    typedef typename cusp::array1d<cusp::complex<float>, MemorySpace>       Array;
    typedef typename cusp::array1d<cusp::complex<float>, MemorySpace>::view View;

    Array x(6);
    Array y(6);

    x[0] = cusp::complex<float>( 7.0f, 0.0f);
    y[0] = cusp::complex<float>( 0.0f, 0.0f);

    x[1] = cusp::complex<float>( 5.0f, 0.0f);
    y[1] = cusp::complex<float>(-2.0f, 0.0f);

    x[2] = cusp::complex<float>( 4.0f, 0.0f);
    y[2] = cusp::complex<float>( 0.0f, 0.0f);

    x[3] = cusp::complex<float>(-3.0f, 0.0f);
    y[3] = cusp::complex<float>( 5.0f, 0.0f);

    x[4] = cusp::complex<float>( 0.0f, 0.0f);
    y[4] = cusp::complex<float>( 6.0f, 0.0f);

    x[5] = cusp::complex<float>( 4.0f, 0.0f);
    y[5] = cusp::complex<float>( 1.0f, 0.0f);

    ASSERT_EQUAL(cusp::blas::dotc(x, y), -21.0f);

    ASSERT_EQUAL(cusp::blas::dotc(View(x), View(y)), -21.0f);

    // test size checking
    Array w(3);
    ASSERT_THROWS(cusp::blas::dotc(x, w), cusp::invalid_input_exception);
}
DECLARE_HOST_DEVICE_UNITTEST(TestDotc);


template <class MemorySpace>
void TestFill(void)
{
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    Array x(4);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;

    cusp::blas::fill(x, 2.0f);

    ASSERT_EQUAL(x[0], 2.0);
    ASSERT_EQUAL(x[1], 2.0);
    ASSERT_EQUAL(x[2], 2.0);
    ASSERT_EQUAL(x[3], 2.0);

    cusp::blas::fill(View(x), 1.0f);

    ASSERT_EQUAL(x[0], 1.0);
    ASSERT_EQUAL(x[1], 1.0);
    ASSERT_EQUAL(x[2], 1.0);
    ASSERT_EQUAL(x[3], 1.0);
}
DECLARE_HOST_DEVICE_UNITTEST(TestFill);


template <class MemorySpace>
void TestNrm1(void)
{
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    Array x(6);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::nrm1(x), 20.0f);

    ASSERT_EQUAL(cusp::blas::nrm1(View(x)), 20.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestNrm1);

template <class MemorySpace>
void TestComplexNrm1(void)
{
    typedef typename cusp::array1d<cusp::complex<float>, MemorySpace>       Array;
    typedef typename cusp::array1d<cusp::complex<float>, MemorySpace>::view View;

    Array x(6);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::nrm1(x), 20.0f);

    ASSERT_EQUAL(cusp::blas::nrm1(View(x)), 20.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestComplexNrm1);


template <class MemorySpace>
void TestNrm2(void)
{
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    Array x(6);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::nrm2(x), 10.0f);

    ASSERT_EQUAL(cusp::blas::nrm2(View(x)), 10.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestNrm2);

template <class MemorySpace>
void TestComplexNrm2(void)
{
    typedef typename cusp::array1d<cusp::complex<float>, MemorySpace>       Array;
    typedef typename cusp::array1d<cusp::complex<float>, MemorySpace>::view View;

    Array x(6);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::nrm2(x), 10.0f);

    ASSERT_EQUAL(cusp::blas::nrm2(View(x)), 10.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestComplexNrm2);


template <class MemorySpace>
void TestNrmmax(void)
{
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    Array x(6);
    View view_x(x);

    x[0] =  7.0f;
    x[1] = -5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::nrmmax(x), 7.0f);

    ASSERT_EQUAL(cusp::blas::nrmmax(View(x)), 7.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestNrmmax);

template <class MemorySpace>
void TestComplexNrmmax(void)
{
    typedef typename cusp::array1d<cusp::complex<float>, MemorySpace>       Array;
    typedef typename cusp::array1d<cusp::complex<float>, MemorySpace>::view View;

    Array x(6);
    View view_x(x);

    x[0] =  7.0f;
    x[1] = -5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::nrmmax(x), 7.0f);

    ASSERT_EQUAL(cusp::blas::nrmmax(View(x)), 7.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestComplexNrmmax);


template <class MemorySpace>
void TestScal(void)
{
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    Array x(6);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  4.0f;

    cusp::blas::scal(x, 4.0f);

    ASSERT_EQUAL(x[0],  28.0);
    ASSERT_EQUAL(x[1],  20.0);
    ASSERT_EQUAL(x[2],  16.0);
    ASSERT_EQUAL(x[3], -12.0);
    ASSERT_EQUAL(x[4],   0.0);
    ASSERT_EQUAL(x[5],  16.0);

    cusp::blas::scal(View(x), 2.0f);

    ASSERT_EQUAL(x[0],  56.0);
    ASSERT_EQUAL(x[1],  40.0);
    ASSERT_EQUAL(x[2],  32.0);
    ASSERT_EQUAL(x[3], -24.0);
    ASSERT_EQUAL(x[4],   0.0);
    ASSERT_EQUAL(x[5],  32.0);
}
DECLARE_HOST_DEVICE_UNITTEST(TestScal);

