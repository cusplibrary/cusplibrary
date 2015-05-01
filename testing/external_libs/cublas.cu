#include <unittest/unittest.h>
#include <cusp/array2d.h>
#include <cusp/gallery/poisson.h>
#include <cusp/blas/cublas/blas.h>

void TestAxpy(void)
{
    typedef cusp::host_memory MemorySpace;
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
DECLARE_UNITTEST(TestAxpy);
