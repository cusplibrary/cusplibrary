#include <unittest/unittest.h>
#include <cusp/array2d.h>
#include <cusp/gallery/poisson.h>
#include <cusp/blas/cblas/blas.h>
#include <cusp/blas/blas.h>

/* void TestCBLASamax(void) */
/* { */
/*     typedef cusp::host_memory MemorySpace; */
/*     typedef typename cusp::array1d<float, MemorySpace>       Array; */
/*     typedef typename cusp::array1d<float, MemorySpace>::view View; */
/*  */
/*     cusp::cblas::execution_policy cblas; */
/*  */
/*     Array x(6); */
/*     View view_x(x); */
/*  */
/*     x[0] =  7.0f; */
/*     x[1] = -5.0f; */
/*     x[2] =  4.0f; */
/*     x[3] = -3.0f; */
/*     x[4] =  0.0f; */
/*     x[5] =  1.0f; */
/*  */
/*     ASSERT_EQUAL(cusp::blas::amax(cblas,x), 0); */
/*  */
/*     ASSERT_EQUAL(cusp::blas::amax(cblas,view_x), 0); */
/* } */
/* DECLARE_UNITTEST(TestCBLASamax); */
/*  */
/* void TestCBLASasum(void) */
/* { */
/*     typedef cusp::host_memory MemorySpace; */
/*     typedef typename cusp::array1d<float, MemorySpace>       Array; */
/*     typedef typename cusp::array1d<float, MemorySpace>::view View; */
/*  */
/*     cusp::cblas::execution_policy cblas; */
/*  */
/*     Array x(6); */
/*     View view_x(x); */
/*  */
/*     x[0] =  7.0f; */
/*     x[1] =  5.0f; */
/*     x[2] =  4.0f; */
/*     x[3] = -3.0f; */
/*     x[4] =  0.0f; */
/*     x[5] =  1.0f; */
/*  */
/*     ASSERT_EQUAL(cusp::blas::asum(cblas,x), 20.0f); */
/*  */
/*     ASSERT_EQUAL(cusp::blas::asum(cblas,view_x), 20.0f); */
/* } */
/* DECLARE_UNITTEST(TestCBLASasum); */

void TestCBLASaxpy(void)
{
    typedef cusp::host_memory MemorySpace;
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    cusp::cblas::execution_policy cblas;

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

    cusp::blas::axpy(cblas, x, y, 2.0f);

    ASSERT_EQUAL(y[0],  14.0);
    ASSERT_EQUAL(y[1],   8.0);
    ASSERT_EQUAL(y[2],   8.0);
    ASSERT_EQUAL(y[3],  -1.0);

    View view_x(x);
    View view_y(y);

    cusp::blas::axpy(cblas, view_x, view_y, 2.0f);

    ASSERT_EQUAL(y[0],  28.0);
    ASSERT_EQUAL(y[1],  18.0);
    ASSERT_EQUAL(y[2],  16.0);
    ASSERT_EQUAL(y[3],  -7.0);
}
DECLARE_UNITTEST(TestCBLASaxpy);

void TestCBLAScopy(void)
{
    typedef cusp::host_memory MemorySpace;
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    cusp::cblas::execution_policy cblas;

    Array x(4);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;

    {
        Array y(4, -1);
        cusp::blas::copy(cblas, x, y);
        ASSERT_EQUAL(x, y);
    }

    {
        Array y(4, -1);
        View view_x(x);
        View view_y(y);
        cusp::blas::copy(cblas, view_x, view_y);
        ASSERT_EQUAL(x, y);
    }
}
DECLARE_UNITTEST(TestCBLAScopy);

void TestCBLASdot(void)
{
    typedef cusp::host_memory MemorySpace;
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    cusp::cblas::execution_policy cblas;

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

    ASSERT_EQUAL(cusp::blas::dot(cblas, x, y), -21.0f);

    View view_x(x);
    View view_y(y);
    ASSERT_EQUAL(cusp::blas::dot(cblas, view_x, view_y), -21.0f);
}
DECLARE_UNITTEST(TestCBLASdot);

void TestCBLASnrm2(void)
{
    typedef cusp::host_memory MemorySpace;
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    cusp::cblas::execution_policy cblas;

    Array x(6);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  1.0f;

    ASSERT_EQUAL(cusp::blas::nrm2(cblas, x), 10.0f);

    ASSERT_EQUAL(cusp::blas::nrm2(cblas, View(x)), 10.0f);
}
DECLARE_UNITTEST(TestCBLASnrm2);

void TestCBLASscal(void)
{
    typedef cusp::host_memory MemorySpace;
    typedef typename cusp::array1d<float, MemorySpace>       Array;
    typedef typename cusp::array1d<float, MemorySpace>::view View;

    cusp::cblas::execution_policy cblas;

    Array x(6);

    x[0] =  7.0f;
    x[1] =  5.0f;
    x[2] =  4.0f;
    x[3] = -3.0f;
    x[4] =  0.0f;
    x[5] =  4.0f;

    cusp::blas::scal(cblas, x, 4.0f);

    ASSERT_EQUAL(x[0],  28.0);
    ASSERT_EQUAL(x[1],  20.0);
    ASSERT_EQUAL(x[2],  16.0);
    ASSERT_EQUAL(x[3], -12.0);
    ASSERT_EQUAL(x[4],   0.0);
    ASSERT_EQUAL(x[5],  16.0);

    View v(x);
    cusp::blas::scal(cblas, v, 2.0f);

    ASSERT_EQUAL(x[0],  56.0);
    ASSERT_EQUAL(x[1],  40.0);
    ASSERT_EQUAL(x[2],  32.0);
    ASSERT_EQUAL(x[3], -24.0);
    ASSERT_EQUAL(x[4],   0.0);
    ASSERT_EQUAL(x[5],  32.0);
}
DECLARE_UNITTEST(TestCBLASscal);

