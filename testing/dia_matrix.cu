#include <unittest/unittest.h>
#include <cusp/dia_matrix.h>

template <class MemorySpace>
void TestAllocateDiaMatrix(void)
{
    cusp::dia_matrix<int, float, MemorySpace> matrix;

    cusp::allocate_matrix(matrix, 10, 10, 100, 19, 32);
    
    ASSERT_EQUAL(matrix.num_rows,       10);
    ASSERT_EQUAL(matrix.num_cols,       10);
    ASSERT_EQUAL(matrix.num_entries,    100);
    ASSERT_EQUAL(matrix.num_diagonals,  19);
    ASSERT_EQUAL(matrix.stride,         32);
    
    cusp::deallocate_matrix(matrix);

    ASSERT_EQUAL(matrix.num_rows,       0);
    ASSERT_EQUAL(matrix.num_cols,       0);
    ASSERT_EQUAL(matrix.num_entries,    0);
    ASSERT_EQUAL(matrix.num_diagonals,  0);
    ASSERT_EQUAL(matrix.stride,         0);
    ASSERT_EQUAL(matrix.diagonal_offsets, (void *) 0);
    ASSERT_EQUAL(matrix.values,           (void *) 0);
}
DECLARE_HOST_DEVICE_UNITTEST(TestAllocateDiaMatrix);


void TestAllocateDiaMatrixLike(void)
{
    cusp::dia_matrix<int, float, cusp::host_memory> example;
    cusp::dia_matrix<int, float, cusp::host_memory> matrix;

    cusp::allocate_matrix(example, 10, 10, 100, 19, 32);
    cusp::allocate_matrix_like(matrix, example);
    
    ASSERT_EQUAL(matrix.num_rows,       10);
    ASSERT_EQUAL(matrix.num_cols,       10);
    ASSERT_EQUAL(matrix.num_entries,    100);
    ASSERT_EQUAL(matrix.num_diagonals,  19);
    ASSERT_EQUAL(matrix.stride,         32);
    
    cusp::deallocate_matrix(example);
    cusp::deallocate_matrix(matrix);
}
DECLARE_UNITTEST(TestAllocateDiaMatrixLike);


void TestMemcpyDiaMatrix(void)
{
    cusp::dia_matrix<int, float, cusp::host_memory> h1;
    cusp::dia_matrix<int, float, cusp::host_memory> h2;
    cusp::dia_matrix<int, float, cusp::device_memory> d1;
    cusp::dia_matrix<int, float, cusp::device_memory> d2;

    cusp::allocate_matrix(h1, 2, 2, 4, 3, 2);
    cusp::allocate_matrix(h2, 2, 2, 4, 3, 2);
    cusp::allocate_matrix(d1, 2, 2, 4, 3, 2);
    cusp::allocate_matrix(d2, 2, 2, 4, 3, 2);

    // initialize host matrix
    h1.diagonal_offsets[0] = -1;
    h1.diagonal_offsets[1] =  0;
    h1.diagonal_offsets[2] =  1;

    h1.values[0] =  0; 
    h1.values[1] = 12;
    h1.values[2] = 10;
    h1.values[3] = 13;
    h1.values[4] = 11;
    h1.values[5] =  0;

    // memcpy h1 -> d1 -> d2 -> h2
    cusp::memcpy_matrix(d1, h1);
    cusp::memcpy_matrix(d2, d1);
    cusp::memcpy_matrix(h2, d2);

    // compare h1 and h2
    ASSERT_EQUAL(h1.num_rows,    h2.num_rows);
    ASSERT_EQUAL(h1.num_cols,    h2.num_cols);
    ASSERT_EQUAL(h1.num_entries, h2.num_entries);
    ASSERT_EQUAL(h1.num_diagonals, h2.num_diagonals);
    ASSERT_EQUAL(h1.stride,        h2.stride);
    ASSERT_EQUAL_RANGES(h1.diagonal_offsets, h1.diagonal_offsets + 3, h2.diagonal_offsets);
    ASSERT_EQUAL_RANGES(h1.values,           h1.values + 6,           h2.values);

    // change h2
    h2.diagonal_offsets[0] = -1;
    h2.diagonal_offsets[1] =  0;
    h2.diagonal_offsets[2] =  1;

    h2.values[0] =  0; 
    h2.values[1] = 12;
    h2.values[2] = 10;
    h2.values[3] = 13;
    h2.values[3] = 11;
    h2.values[3] =  0;
   
    // memcpy h2 -> h1
    cusp::memcpy_matrix(h2, h1);
    
    // compare h1 and h2 again
    ASSERT_EQUAL(h1.num_rows,    h2.num_rows);
    ASSERT_EQUAL(h1.num_cols,    h2.num_cols);
    ASSERT_EQUAL(h1.num_entries, h2.num_entries);
    ASSERT_EQUAL(h1.num_diagonals, h2.num_diagonals);
    ASSERT_EQUAL(h1.stride,        h2.stride);
    ASSERT_EQUAL_RANGES(h1.diagonal_offsets, h1.diagonal_offsets + 3, h2.diagonal_offsets);
    ASSERT_EQUAL_RANGES(h1.values,           h1.values + 6,           h2.values);

    cusp::deallocate_matrix(h1);
    cusp::deallocate_matrix(h2);
    cusp::deallocate_matrix(d1);
    cusp::deallocate_matrix(d2);
}
DECLARE_UNITTEST(TestMemcpyDiaMatrix);

