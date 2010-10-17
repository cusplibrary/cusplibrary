#include <unittest/unittest.h>

#include <cusp/detail/device/generalized_spmv/csr_scalar.h>
#include <cusp/detail/device/generalized_spmv/coo_flat.h>

#include <cusp/csr_matrix.h>
#include <cusp/array1d.h>
#include <cusp/gallery/poisson.h>

void TestCsrGeneralizedSpMV(void)
{
    typedef cusp::csr_matrix<int,float,cusp::device_memory> TestMatrix;
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

    cusp::detail::device::cuda::spmv_csr_scalar
        (test_matrix.num_rows,
         test_matrix.row_offsets.begin(), test_matrix.column_indices.begin(), test_matrix.values.begin(),
         x.begin(), y.begin(), y.begin(),
         thrust::identity<float>(), thrust::multiplies<float>(), thrust::plus<float>());
                                               
    ASSERT_EQUAL(y[0], 183.0f);
    ASSERT_EQUAL(y[1],  74.0f);
    ASSERT_EQUAL(y[2], 325.0f);
    ASSERT_EQUAL(y[3], 510.0f);
    ASSERT_EQUAL(y[4], 131.0f);
}
DECLARE_UNITTEST(TestCsrGeneralizedSpMV);

void TestCooGeneralizedSpMV(void)
{
    typedef cusp::coo_matrix<int,float,cusp::device_memory> TestMatrix;
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

    cusp::detail::device::cuda::spmv_coo
        (test_matrix.num_rows, test_matrix.num_entries,
         test_matrix.row_indices.begin(), test_matrix.column_indices.begin(), test_matrix.values.begin(),
         x.begin(), y.begin(), y.begin(),
         thrust::identity<float>(), thrust::multiplies<float>(), thrust::plus<float>());
                                               
    ASSERT_EQUAL(y[0], 183.0f);
    ASSERT_EQUAL(y[1],  74.0f);
    ASSERT_EQUAL(y[2], 325.0f);
    ASSERT_EQUAL(y[3], 510.0f);
    ASSERT_EQUAL(y[4], 131.0f);
}
DECLARE_UNITTEST(TestCooGeneralizedSpMV);


