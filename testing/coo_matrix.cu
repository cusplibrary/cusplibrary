#include <unittest/unittest.h>
#include <cusp/coo_matrix.h>

void TestAllocateCooPattern(void)
{
    cusp::coo_pattern<int, cusp::host_memory> pattern;

    cusp::allocate_pattern(pattern, 10, 10, 100);
    
    ASSERT_EQUAL(pattern.num_rows,       10);
    ASSERT_EQUAL(pattern.num_cols,       10);
    ASSERT_EQUAL(pattern.num_entries,    100);
    
    cusp::deallocate_pattern(pattern);

    ASSERT_EQUAL(pattern.num_rows,       0);
    ASSERT_EQUAL(pattern.num_cols,       0);
    ASSERT_EQUAL(pattern.num_entries,    0);
    ASSERT_EQUAL(pattern.row_indices,    (void *) 0);
    ASSERT_EQUAL(pattern.column_indices, (void *) 0);
}
DECLARE_UNITTEST(TestAllocateCooPattern);


void TestAllocateCooPatternLike(void)
{
    cusp::coo_pattern<int, cusp::host_memory> pattern;
    cusp::coo_pattern<int, cusp::host_memory> example;

    cusp::allocate_pattern(example, 10, 10, 100);
    cusp::allocate_pattern_like(pattern, example);
    
    ASSERT_EQUAL(pattern.num_rows,       10);
    ASSERT_EQUAL(pattern.num_cols,       10);
    ASSERT_EQUAL(pattern.num_entries,    100);

    cusp::deallocate_pattern(pattern);
    cusp::deallocate_pattern(example);
}
DECLARE_UNITTEST(TestAllocateCooPatternLike);


template <class MemorySpace>
void TestAllocateCooMatrix(void)
{
    cusp::coo_matrix<int, float, MemorySpace> matrix;

    cusp::allocate_matrix(matrix, 10, 10, 100);
    
    ASSERT_EQUAL(matrix.num_rows,       10);
    ASSERT_EQUAL(matrix.num_cols,       10);
    ASSERT_EQUAL(matrix.num_entries,    100);
    
    cusp::deallocate_matrix(matrix);

    ASSERT_EQUAL(matrix.num_rows,       0);
    ASSERT_EQUAL(matrix.num_cols,       0);
    ASSERT_EQUAL(matrix.num_entries,    0);
    ASSERT_EQUAL(matrix.row_indices,    (void *) 0);
    ASSERT_EQUAL(matrix.column_indices, (void *) 0);
    ASSERT_EQUAL(matrix.values,         (void *) 0);
}
DECLARE_HOST_DEVICE_UNITTEST(TestAllocateCooMatrix);


void TestAllocateCooMatrixLike(void)
{
    cusp::coo_matrix<int, float, cusp::host_memory> matrix;
    cusp::coo_matrix<int, float, cusp::host_memory> example;
    
    cusp::allocate_matrix(example, 10, 10, 100);
    cusp::allocate_matrix_like(matrix, example);
    
    ASSERT_EQUAL(matrix.num_rows,       10);
    ASSERT_EQUAL(matrix.num_cols,       10);
    ASSERT_EQUAL(matrix.num_entries,    100);
    
    cusp::deallocate_matrix(matrix);
    cusp::deallocate_matrix(example);
}
DECLARE_UNITTEST(TestAllocateCooMatrixLike);


void TestMemcpyCooMatrix(void)
{
    cusp::coo_matrix<int, float, cusp::host_memory> h1;
    cusp::coo_matrix<int, float, cusp::host_memory> h2;
    cusp::coo_matrix<int, float, cusp::device_memory> d1;
    cusp::coo_matrix<int, float, cusp::device_memory> d2;

    cusp::allocate_matrix(h1, 2, 2, 4);
    cusp::allocate_matrix(h2, 2, 2, 4);
    cusp::allocate_matrix(d1, 2, 2, 4);
    cusp::allocate_matrix(d2, 2, 2, 4);

    // initialize host matrix
    h1.row_indices[0] = 0;   h1.column_indices[0] = 0;   h1.values[0] = 10; 
    h1.row_indices[1] = 0;   h1.column_indices[1] = 1;   h1.values[1] = 11;
    h1.row_indices[2] = 1;   h1.column_indices[2] = 0;   h1.values[2] = 12;
    h1.row_indices[3] = 1;   h1.column_indices[3] = 1;   h1.values[3] = 13;

    // memcpy h1 -> d1 -> d2 -> h2
    cusp::memcpy_matrix(d1, h1);
    cusp::memcpy_matrix(d2, d1);
    cusp::memcpy_matrix(h2, d2);

    // compare h1 and h2
    ASSERT_EQUAL(h1.num_rows,    h2.num_rows);
    ASSERT_EQUAL(h1.num_cols,    h2.num_cols);
    ASSERT_EQUAL(h1.num_entries, h2.num_entries);
    ASSERT_EQUAL_RANGES(h1.row_indices,    h1.row_indices + 4,    h2.row_indices);
    ASSERT_EQUAL_RANGES(h1.column_indices, h1.column_indices + 4, h2.column_indices);
    ASSERT_EQUAL_RANGES(h1.values, h1.values + 4, h2.values);

    // change h2
    h1.row_indices[0] = 0;   h1.column_indices[0] = 0;   h1.values[0] = 0; 
    h1.row_indices[1] = 0;   h1.column_indices[1] = 0;   h1.values[1] = 0;
    h1.row_indices[2] = 0;   h1.column_indices[2] = 0;   h1.values[2] = 0;
    h1.row_indices[3] = 0;   h1.column_indices[3] = 0;   h1.values[3] = 0;
   
    // memcpy h2 -> h1
    cusp::memcpy_matrix(h2, h1);
    
    // compare h1 and h2 again
    ASSERT_EQUAL(h1.num_rows,    h2.num_rows);
    ASSERT_EQUAL(h1.num_cols,    h2.num_cols);
    ASSERT_EQUAL(h1.num_entries, h2.num_entries);
    ASSERT_EQUAL_RANGES(h1.row_indices,    h1.row_indices + 4,    h2.row_indices);
    ASSERT_EQUAL_RANGES(h1.column_indices, h1.column_indices + 4, h2.column_indices);
    ASSERT_EQUAL_RANGES(h1.values, h1.values + 4, h2.values);

    cusp::deallocate_matrix(h1);
    cusp::deallocate_matrix(h2);
    cusp::deallocate_matrix(d1);
    cusp::deallocate_matrix(d2);
}
DECLARE_UNITTEST(TestMemcpyCooMatrix);

