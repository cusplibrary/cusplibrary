#include <unittest/unittest.h>

#define CUSP_USE_TEXTURE_MEMORY

#include <cusp/csr_matrix.h>
#include <cusp/array1d.h>
#include <cusp/gallery/poisson.h>

template <class TestMatrix>
void TestSpMV()
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
    TestMatrix test_matrix = A;

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
DECLARE_SPARSE_MATRIX_UNITTEST(TestSpMV);


template <class TestMatrix>
void TestSpMVTextureCache()
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
        TestMatrix test_matrix = A;

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
        TestMatrix test_matrix;
        cusp::gallery::poisson5pt(test_matrix, 10, 10);

        // allocate vectors
        cusp::array1d<float, MemorySpace> x(test_matrix.num_cols + 1); // offset by one
        cusp::array1d<float, MemorySpace> y(test_matrix.num_rows);

        ASSERT_THROWS(cusp::detail::device::spmv_tex(test_matrix, thrust::raw_pointer_cast(&x[0]) + 1, thrust::raw_pointer_cast(&y[0])),
                      cusp::invalid_input_exception);

    }
}
DECLARE_DEVICE_SPARSE_MATRIX_UNITTEST(TestSpMVTextureCache);

