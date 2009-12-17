#include <unittest/unittest.h>

#include <cusp/csr_matrix.h>
#include <cusp/array1d.h>
#include <cusp/gallery/poisson.h>

template <class TestMatrix>
void _TestSpMV(TestMatrix test_matrix)
{
    typedef typename TestMatrix::memory_space MemorySpace;

    // initialize example matrix
    cusp::array2d<float, cusp::host_memory> A(5,4);
    A(0,0) = 13; A(0,1) = 80; A(0,2) =  0; A(0,3) =  0; 
    A(1,0) =  0; A(1,1) = 27; A(1,2) =  0; A(1,3) =  0;
    A(2,0) = 55; A(2,1) =  0; A(2,2) = 24; A(2,3) = 42;
    A(3,0) =  0; A(3,1) = 69; A(3,2) =  0; A(3,3) = 83;
    A(4,0) =  0; A(4,1) =  0; A(4,2) = 27; A(4,3) =  0;

    // convert to desired format
    test_matrix = A;

    // allocate vectors
    cusp::array1d<float, MemorySpace> x(4);
    cusp::array1d<float, MemorySpace> y(5);
   
    // initialize input and output vectors
    x[0] = 1.0f; y[0] = 10.0f; 
    x[1] = 2.0f; y[1] = 20.0f;
    x[2] = 3.0f; y[2] = 30.0f;
    x[3] = 4.0f; y[3] = 40.0f;
                 y[4] = 50.0f;

    test_matrix.multiply(x, y);

    ASSERT_EQUAL(y[0], 183.0f);
    ASSERT_EQUAL(y[1],  74.0f);
    ASSERT_EQUAL(y[2], 325.0f);
    ASSERT_EQUAL(y[3], 510.0f);
    ASSERT_EQUAL(y[4], 131.0f);
}

template <class MemorySpace>
void TestCooSpMV(void)
{   _TestSpMV(cusp::coo_matrix<int, float, MemorySpace>());  }
DECLARE_HOST_DEVICE_UNITTEST(TestCooSpMV);

template <class MemorySpace>
void TestCsrSpMV(void)
{   _TestSpMV(cusp::csr_matrix<int, float, MemorySpace>());  }
DECLARE_HOST_DEVICE_UNITTEST(TestCsrSpMV);

template <class MemorySpace>
void TestDiaSpMV(void)
{   _TestSpMV(cusp::dia_matrix<int, float, MemorySpace>());  }
DECLARE_HOST_DEVICE_UNITTEST(TestDiaSpMV);

template <class MemorySpace>
void TestEllSpMV(void)
{   _TestSpMV(cusp::ell_matrix<int, float, MemorySpace>());  }
DECLARE_HOST_DEVICE_UNITTEST(TestEllSpMV);

template <class MemorySpace>
void TestHybSpMV(void)
{   _TestSpMV(cusp::hyb_matrix<int, float, MemorySpace>());  }
DECLARE_HOST_DEVICE_UNITTEST(TestHybSpMV);


template <class TestMatrix>
void _TestSpMVTextureCache(TestMatrix test_matrix)
{
    typedef typename TestMatrix::memory_space MemorySpace;

    // test with aligned memory
    {
        // initialize example matrix
        cusp::array2d<float, cusp::host_memory> A(5,4);
        A(0,0) = 13; A(0,1) = 80; A(0,2) =  0; A(0,3) =  0; 
        A(1,0) =  0; A(1,1) = 27; A(1,2) =  0; A(1,3) =  0;
        A(2,0) = 55; A(2,1) =  0; A(2,2) = 24; A(2,3) = 42;
        A(3,0) =  0; A(3,1) = 69; A(3,2) =  0; A(3,3) = 83;
        A(4,0) =  0; A(4,1) =  0; A(4,2) = 27; A(4,3) =  0;

        // convert to desired format
        test_matrix = A;

        // allocate vectors
        cusp::array1d<float, MemorySpace> x(4);
        cusp::array1d<float, MemorySpace> y(5);

        // initialize input and output vectors
        x[0] = 1.0f; y[0] = 10.0f; 
        x[1] = 2.0f; y[1] = 20.0f;
        x[2] = 3.0f; y[2] = 30.0f;
        x[3] = 4.0f; y[3] = 40.0f;
                     y[4] = 50.0f;

        cusp::detail::device::spmv_tex(test_matrix,
                                       thrust::raw_pointer_cast(&x[0]),
                                       thrust::raw_pointer_cast(&y[0]));

        ASSERT_EQUAL(y[0], 183.0f);
        ASSERT_EQUAL(y[1],  74.0f);
        ASSERT_EQUAL(y[2], 325.0f);
        ASSERT_EQUAL(y[3], 510.0f);
        ASSERT_EQUAL(y[4], 131.0f);
    }
    
    // test with unaligned memory
    {
        cusp::gallery::poisson5pt(test_matrix, 10, 10);

        // allocate vectors
        cusp::array1d<float, MemorySpace> x(test_matrix.num_cols + 1); // offset by one
        cusp::array1d<float, MemorySpace> y(test_matrix.num_rows);

        ASSERT_THROWS(cusp::detail::device::spmv_tex(test_matrix, thrust::raw_pointer_cast(&x[0]) + 1, thrust::raw_pointer_cast(&y[0])),
                      cusp::invalid_input_exception);

    }
}

void TestCooSpMVTextureCache(void)
{   _TestSpMVTextureCache(cusp::coo_matrix<int, float, cusp::device_memory>());  }
DECLARE_UNITTEST(TestCooSpMVTextureCache);

void TestCsrSpMVTextureCache(void)
{   _TestSpMVTextureCache(cusp::csr_matrix<int, float, cusp::device_memory>());  }
DECLARE_UNITTEST(TestCsrSpMVTextureCache);

void TestDiaSpMVTextureCache(void)
{   _TestSpMVTextureCache(cusp::dia_matrix<int, float, cusp::device_memory>());  }
DECLARE_UNITTEST(TestDiaSpMVTextureCache);

void TestEllSpMVTextureCache(void)
{   _TestSpMVTextureCache(cusp::ell_matrix<int, float, cusp::device_memory>());  }
DECLARE_UNITTEST(TestEllSpMVTextureCache);

void TestHybSpMVTextureCache(void)
{   _TestSpMVTextureCache(cusp::hyb_matrix<int, float, cusp::device_memory>());  }
DECLARE_UNITTEST(TestHybSpMVTextureCache);

