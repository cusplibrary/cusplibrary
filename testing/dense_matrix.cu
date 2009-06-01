#include <unittest/unittest.h>
#include <cusp/dense_matrix.h>

template <class MemorySpace>
void TestAllocateDenseMatrix(void)
{
    cusp::dense_matrix<float, MemorySpace> matrix;

    cusp::allocate_matrix(matrix, 10, 10);
    
    ASSERT_EQUAL(matrix.num_rows,       10);
    ASSERT_EQUAL(matrix.num_cols,       10);
    ASSERT_EQUAL(matrix.num_entries,    100);

    cusp::deallocate_matrix(matrix);

    ASSERT_EQUAL(matrix.num_rows,       0);
    ASSERT_EQUAL(matrix.num_cols,       0);
    ASSERT_EQUAL(matrix.num_entries,    0);
    ASSERT_EQUAL(matrix.values,         (void *) 0);
}
DECLARE_HOST_DEVICE_UNITTEST(TestAllocateDenseMatrix);


void TestMemcpyDenseMatrix(void)
{
    cusp::dense_matrix<float, cusp::host_memory> h1;
    cusp::dense_matrix<float, cusp::host_memory> h2;
    cusp::dense_matrix<float, cusp::device_memory> d1;
    cusp::dense_matrix<float, cusp::device_memory> d2;

    cusp::allocate_matrix(h1, 2, 2);
    cusp::allocate_matrix(h2, 2, 2);
    cusp::allocate_matrix(d1, 2, 2);
    cusp::allocate_matrix(d2, 2, 2);

    // initialize host matrix
    h1.values[0] = 10; 
    h1.values[1] = 11;
    h1.values[2] = 12;
    h1.values[3] = 13;

    // memcpy h1 -> d1 -> d2 -> h2
    cusp::memcpy_matrix(d1, h1);
    cusp::memcpy_matrix(d2, d1);
    cusp::memcpy_matrix(h2, d2);

    // compare h1 and h2
    ASSERT_EQUAL(h1.num_rows,    h2.num_rows);
    ASSERT_EQUAL(h1.num_cols,    h2.num_cols);
    ASSERT_EQUAL(h1.num_entries, h2.num_entries);
    ASSERT_EQUAL_RANGES(h1.values, h1.values + 4, h2.values);

    // change h2
    h1.values[0] = 0; 
    h1.values[1] = 0;
    h1.values[2] = 0;
    h1.values[3] = 0;
   
    // memcpy h2 -> h1
    cusp::memcpy_matrix(h2, h1);
    
    // compare h1 and h2 again
    ASSERT_EQUAL(h1.num_rows,    h2.num_rows);
    ASSERT_EQUAL(h1.num_cols,    h2.num_cols);
    ASSERT_EQUAL(h1.num_entries, h2.num_entries);
    ASSERT_EQUAL_RANGES(h1.values, h1.values + 4, h2.values);

    cusp::deallocate_matrix(h1);
    cusp::deallocate_matrix(h2);
    cusp::deallocate_matrix(d1);
    cusp::deallocate_matrix(d2);
}
DECLARE_UNITTEST(TestMemcpyDenseMatrix);


void TestDenseMatrixRowMajor(void)
{
    cusp::dense_matrix<float, cusp::host_memory, cusp::row_major> matrix;

    cusp::allocate_matrix(matrix, 2, 2);
    
    matrix(0,0) = 10;  matrix(0,1) = 20;
    matrix(1,0) = 30;  matrix(1,1) = 40;

    ASSERT_EQUAL(matrix(0,0), 10);    ASSERT_EQUAL(matrix(0,1), 20);
    ASSERT_EQUAL(matrix(1,0), 30);    ASSERT_EQUAL(matrix(1,1), 40);

    ASSERT_EQUAL(matrix.values[0], 10);
    ASSERT_EQUAL(matrix.values[1], 20);
    ASSERT_EQUAL(matrix.values[2], 30);
    ASSERT_EQUAL(matrix.values[3], 40);

    cusp::deallocate_matrix(matrix);
}
DECLARE_UNITTEST(TestDenseMatrixRowMajor);


void TestDenseMatrixColumnMajor(void)
{
    cusp::dense_matrix<float, cusp::host_memory, cusp::column_major> matrix;

    cusp::allocate_matrix(matrix, 2, 2);
    
    matrix(0,0) = 10;  matrix(0,1) = 20;
    matrix(1,0) = 30;  matrix(1,1) = 40;

    ASSERT_EQUAL(matrix(0,0), 10);    ASSERT_EQUAL(matrix(0,1), 20);
    ASSERT_EQUAL(matrix(1,0), 30);    ASSERT_EQUAL(matrix(1,1), 40);

    ASSERT_EQUAL(matrix.values[0], 10);
    ASSERT_EQUAL(matrix.values[1], 30);
    ASSERT_EQUAL(matrix.values[2], 20);
    ASSERT_EQUAL(matrix.values[3], 40);

    cusp::deallocate_matrix(matrix);
}
DECLARE_UNITTEST(TestDenseMatrixColumnMajor);


