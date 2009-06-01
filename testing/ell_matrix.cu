#include <unittest/unittest.h>
#include <cusp/ell_matrix.h>

void TestAllocateEllPattern(void)
{
    cusp::ell_pattern<int, cusp::host_memory> pattern;

    cusp::allocate_pattern(pattern, 10, 10, 50, 5, 16);
    
    ASSERT_EQUAL(pattern.num_rows,            10);
    ASSERT_EQUAL(pattern.num_cols,            10);
    ASSERT_EQUAL(pattern.num_entries,         50);
    ASSERT_EQUAL(pattern.num_entries_per_row,  5);
    ASSERT_EQUAL(pattern.stride,              16);
    
    cusp::deallocate_pattern(pattern);

    ASSERT_EQUAL(pattern.num_rows,             0);
    ASSERT_EQUAL(pattern.num_cols,             0);
    ASSERT_EQUAL(pattern.num_entries,          0);
    ASSERT_EQUAL(pattern.num_entries_per_row,  0);
    ASSERT_EQUAL(pattern.stride,               0);
    ASSERT_EQUAL(pattern.column_indices, (void *) 0);
}
DECLARE_UNITTEST(TestAllocateEllPattern);


void TestAllocateEllPatternLike(void)
{
    cusp::ell_pattern<int, cusp::host_memory> pattern;
    cusp::ell_pattern<int, cusp::host_memory> example;

    cusp::allocate_pattern(example, 10, 10, 50, 5, 16);
    cusp::allocate_pattern_like(pattern, example);
    
    ASSERT_EQUAL(pattern.num_rows,            10);
    ASSERT_EQUAL(pattern.num_cols,            10);
    ASSERT_EQUAL(pattern.num_entries,         50);
    ASSERT_EQUAL(pattern.num_entries_per_row,  5);
    ASSERT_EQUAL(pattern.stride,              16);
    
    cusp::deallocate_pattern(pattern);
    cusp::deallocate_pattern(example);
}
DECLARE_UNITTEST(TestAllocateEllPatternLike);


template <class MemorySpace>
void TestAllocateEllMatrix(void)
{
    cusp::ell_matrix<int, float, MemorySpace> matrix;

    cusp::allocate_matrix(matrix, 10, 10, 50, 5, 16);
    
    ASSERT_EQUAL(matrix.num_rows,             10);
    ASSERT_EQUAL(matrix.num_cols,             10);
    ASSERT_EQUAL(matrix.num_entries,          50);
    ASSERT_EQUAL(matrix.num_entries_per_row,   5);
    ASSERT_EQUAL(matrix.stride,               16);
    
    cusp::deallocate_matrix(matrix);

    ASSERT_EQUAL(matrix.num_rows,             0);
    ASSERT_EQUAL(matrix.num_cols,             0);
    ASSERT_EQUAL(matrix.num_entries,          0);
    ASSERT_EQUAL(matrix.num_entries_per_row,  0);
    ASSERT_EQUAL(matrix.stride,               0);
    ASSERT_EQUAL(matrix.column_indices, (void *) 0);
    ASSERT_EQUAL(matrix.values,         (void *) 0);
}
DECLARE_HOST_DEVICE_UNITTEST(TestAllocateEllMatrix);


void TestAllocateEllMatrixLike(void)
{
    cusp::ell_matrix<int, float, cusp::host_memory> pattern;
    cusp::ell_matrix<int, float, cusp::host_memory> example;

    cusp::allocate_matrix(example, 10, 10, 50, 5, 16);
    cusp::allocate_matrix_like(pattern, example);
    
    ASSERT_EQUAL(pattern.num_rows,            10);
    ASSERT_EQUAL(pattern.num_cols,            10);
    ASSERT_EQUAL(pattern.num_entries,         50);
    ASSERT_EQUAL(pattern.num_entries_per_row,  5);
    ASSERT_EQUAL(pattern.stride,              16);
    
    cusp::deallocate_matrix(pattern);
    cusp::deallocate_matrix(example);
}
DECLARE_UNITTEST(TestAllocateEllMatrixLike);


void TestMemcpyEllMatrix(void)
{
    cusp::ell_matrix<int, float, cusp::host_memory> h1;
    cusp::ell_matrix<int, float, cusp::host_memory> h2;
    cusp::ell_matrix<int, float, cusp::device_memory> d1;
    cusp::ell_matrix<int, float, cusp::device_memory> d2;

    cusp::allocate_matrix(h1, 3, 3, 6, 2, 16);
    cusp::allocate_matrix(h2, 3, 3, 6, 2, 16);
    cusp::allocate_matrix(d1, 3, 3, 6, 2, 16);
    cusp::allocate_matrix(d2, 3, 3, 6, 2, 16);

    // initialize host matrix
    std::fill(h1.column_indices, h1.column_indices + (2 * 16), 0);
    std::fill(h1.values,         h1.values         + (2 * 16), 0);

    h1.column_indices[ 0] = 0;   h1.values[ 0] = 10; 
    h1.column_indices[ 1] = 1;   h1.values[ 1] = 11;
    h1.column_indices[ 2] = 0;   h1.values[ 2] = 12;
    h1.column_indices[16] = 1;   h1.values[16] = 13;
    h1.column_indices[17] = 0;   h1.values[17] = 14;
    h1.column_indices[18] = 1;   h1.values[18] = 15;

    // memcpy h1 -> d1 -> d2 -> h2
    cusp::memcpy_matrix(d1, h1);
    cusp::memcpy_matrix(d2, d1);
    cusp::memcpy_matrix(h2, d2);

    // compare h1 and h2
    ASSERT_EQUAL(h1.num_rows,    h2.num_rows);
    ASSERT_EQUAL(h1.num_cols,    h2.num_cols);
    ASSERT_EQUAL(h1.num_entries, h2.num_entries);
    ASSERT_EQUAL_RANGES(h1.column_indices, h1.column_indices + (2 * 16), h2.column_indices);
    ASSERT_EQUAL_RANGES(h1.values,         h1.values         + (2 * 16), h2.values);

    // change h2
    h1.column_indices[ 0] = 1;   h1.values[ 0] = 20; 
    h1.column_indices[ 1] = 2;   h1.values[ 1] = 21;
    h1.column_indices[ 2] = 1;   h1.values[ 2] = 22;
    h1.column_indices[16] = 2;   h1.values[16] = 23;
    h1.column_indices[17] = 1;   h1.values[17] = 24;
    h1.column_indices[18] = 2;   h1.values[18] = 25;
   
    // memcpy h2 -> h1
    cusp::memcpy_matrix(h2, h1);
    
    // compare h1 and h2 again
    ASSERT_EQUAL(h1.num_rows,    h2.num_rows);
    ASSERT_EQUAL(h1.num_cols,    h2.num_cols);
    ASSERT_EQUAL(h1.num_entries, h2.num_entries);
    ASSERT_EQUAL_RANGES(h1.column_indices, h1.column_indices + (2 * 16), h2.column_indices);
    ASSERT_EQUAL_RANGES(h1.values,         h1.values         + (2 * 16), h2.values);

    cusp::deallocate_matrix(h1);
    cusp::deallocate_matrix(h2);
    cusp::deallocate_matrix(d1);
    cusp::deallocate_matrix(d2);
}
DECLARE_UNITTEST(TestMemcpyEllMatrix);

