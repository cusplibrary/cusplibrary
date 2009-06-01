#include <unittest/unittest.h>
#include <cusp/hyb_matrix.h>

template <class MemorySpace>
void TestAllocateHybMatrix(void)
{
    cusp::hyb_matrix<int, float, MemorySpace> matrix;

    cusp::allocate_matrix(matrix, 10, 10, 50, 13, 5, 16);
    
    ASSERT_EQUAL(matrix.num_rows,                 10);
    ASSERT_EQUAL(matrix.num_cols,                 10);
    ASSERT_EQUAL(matrix.num_entries,              63);

    ASSERT_EQUAL(matrix.ell.num_rows,             10);
    ASSERT_EQUAL(matrix.ell.num_cols,             10);
    ASSERT_EQUAL(matrix.ell.num_entries,          50);
    ASSERT_EQUAL(matrix.ell.num_entries_per_row,   5);
    ASSERT_EQUAL(matrix.ell.stride,               16);

    ASSERT_EQUAL(matrix.coo.num_rows,             10);
    ASSERT_EQUAL(matrix.coo.num_cols,             10);
    ASSERT_EQUAL(matrix.coo.num_entries,          13);
    
    cusp::deallocate_matrix(matrix);
    
    ASSERT_EQUAL(matrix.num_rows,                  0);
    ASSERT_EQUAL(matrix.num_cols,                  0);
    ASSERT_EQUAL(matrix.num_entries,               0);

    ASSERT_EQUAL(matrix.ell.num_rows,              0);
    ASSERT_EQUAL(matrix.ell.num_cols,              0);
    ASSERT_EQUAL(matrix.ell.num_entries,           0);
    ASSERT_EQUAL(matrix.ell.num_entries_per_row,   0);
    ASSERT_EQUAL(matrix.ell.stride,                0);
    ASSERT_EQUAL(matrix.ell.column_indices, (void *) 0);
    ASSERT_EQUAL(matrix.ell.values,         (void *) 0);

    ASSERT_EQUAL(matrix.coo.num_rows,              0);
    ASSERT_EQUAL(matrix.coo.num_cols,              0);
    ASSERT_EQUAL(matrix.coo.num_entries,           0);
    ASSERT_EQUAL(matrix.coo.row_indices,    (void *) 0);
    ASSERT_EQUAL(matrix.coo.column_indices, (void *) 0);
    ASSERT_EQUAL(matrix.coo.values,         (void *) 0);
}
DECLARE_HOST_DEVICE_UNITTEST(TestAllocateHybMatrix);


template <class MemorySpace>
void TestAllocateHybMatrixLike(void)
{
    cusp::hyb_matrix<int, float, MemorySpace> example;
    cusp::hyb_matrix<int, float, MemorySpace> matrix;

    cusp::allocate_matrix(example, 10, 10, 50, 13, 5, 16);
    cusp::allocate_matrix_like(matrix, example);
    
    ASSERT_EQUAL(matrix.num_rows,                 10);
    ASSERT_EQUAL(matrix.num_cols,                 10);
    ASSERT_EQUAL(matrix.num_entries,              63);

    ASSERT_EQUAL(matrix.ell.num_rows,             10);
    ASSERT_EQUAL(matrix.ell.num_cols,             10);
    ASSERT_EQUAL(matrix.ell.num_entries,          50);
    ASSERT_EQUAL(matrix.ell.num_entries_per_row,   5);
    ASSERT_EQUAL(matrix.ell.stride,               16);

    ASSERT_EQUAL(matrix.coo.num_rows,             10);
    ASSERT_EQUAL(matrix.coo.num_cols,             10);
    ASSERT_EQUAL(matrix.coo.num_entries,          13);
    
    cusp::deallocate_matrix(matrix);
    cusp::deallocate_matrix(example);
}
DECLARE_HOST_DEVICE_UNITTEST(TestAllocateHybMatrixLike);


void TestMemcpyHybMatrix(void)
{
    cusp::hyb_matrix<int, float, cusp::host_memory> h1;
    cusp::hyb_matrix<int, float, cusp::host_memory> h2;
    cusp::hyb_matrix<int, float, cusp::device_memory> d1;
    cusp::hyb_matrix<int, float, cusp::device_memory> d2;

    // 3x3 matrix with 6 ELL entries (2 per row) and 3 COO entries
    cusp::allocate_matrix(h1, 3, 3, 6, 3, 2, 16);
    cusp::allocate_matrix(h2, 3, 3, 6, 3, 2, 16);
    cusp::allocate_matrix(d1, 3, 3, 6, 3, 2, 16);
    cusp::allocate_matrix(d2, 3, 3, 6, 3, 2, 16);
    
    // initialize host matrix
    std::fill(h1.ell.column_indices, h1.ell.column_indices + (2 * 16), 0);
    std::fill(h1.ell.values,         h1.ell.values         + (2 * 16), 0);

    h1.ell.column_indices[ 0] = 0;   h1.ell.values[ 0] = 10; 
    h1.ell.column_indices[ 1] = 1;   h1.ell.values[ 1] = 11;
    h1.ell.column_indices[ 2] = 0;   h1.ell.values[ 2] = 12;
    h1.ell.column_indices[16] = 1;   h1.ell.values[16] = 13;
    h1.ell.column_indices[17] = 0;   h1.ell.values[17] = 14;
    h1.ell.column_indices[18] = 1;   h1.ell.values[18] = 15;

    h1.coo.row_indices[0] = 0;   h1.coo.column_indices[0] = 2;   h1.coo.values[0] = 10; 
    h1.coo.row_indices[1] = 1;   h1.coo.column_indices[1] = 2;   h1.coo.values[1] = 11;
    h1.coo.row_indices[2] = 2;   h1.coo.column_indices[2] = 2;   h1.coo.values[2] = 12;

    // memcpy h1 -> d1 -> d2 -> h2
    cusp::memcpy_matrix(d1, h1);
    cusp::memcpy_matrix(d2, d1);
    cusp::memcpy_matrix(h2, d2);

    // compare h1 and h2
    ASSERT_EQUAL(h1.num_rows,    h2.num_rows);
    ASSERT_EQUAL(h1.num_cols,    h2.num_cols);
    ASSERT_EQUAL(h1.num_entries, h2.num_entries);
    ASSERT_EQUAL_RANGES(h1.ell.column_indices, h1.ell.column_indices + (2 * 16), h2.ell.column_indices);
    ASSERT_EQUAL_RANGES(h1.ell.values,         h1.ell.values         + (2 * 16), h2.ell.values);
    ASSERT_EQUAL_RANGES(h1.coo.row_indices,    h1.coo.row_indices    + 3, h2.coo.row_indices);
    ASSERT_EQUAL_RANGES(h1.coo.column_indices, h1.coo.column_indices + 3, h2.coo.column_indices);
    ASSERT_EQUAL_RANGES(h1.coo.values,         h1.coo.values         + 3, h2.coo.values);

    // change h2
    h1.ell.column_indices[ 0] = 1;   h1.ell.values[ 0] = 20; 
    h1.ell.column_indices[ 1] = 2;   h1.ell.values[ 1] = 21;
    h1.ell.column_indices[ 2] = 1;   h1.ell.values[ 2] = 22;
    h1.ell.column_indices[16] = 2;   h1.ell.values[16] = 23;
    h1.ell.column_indices[17] = 1;   h1.ell.values[17] = 24;
    h1.ell.column_indices[18] = 2;   h1.ell.values[18] = 25;
   
    // memcpy h2 -> h1
    cusp::memcpy_matrix(h2, h1);
    
    // compare h1 and h2 again
    ASSERT_EQUAL(h1.num_rows,    h2.num_rows);
    ASSERT_EQUAL(h1.num_cols,    h2.num_cols);
    ASSERT_EQUAL(h1.num_entries, h2.num_entries);
    ASSERT_EQUAL_RANGES(h1.ell.column_indices, h1.ell.column_indices + (2 * 16), h2.ell.column_indices);
    ASSERT_EQUAL_RANGES(h1.ell.values,         h1.ell.values         + (2 * 16), h2.ell.values);
    ASSERT_EQUAL_RANGES(h1.coo.row_indices,    h1.coo.row_indices    + 3, h2.coo.row_indices);
    ASSERT_EQUAL_RANGES(h1.coo.column_indices, h1.coo.column_indices + 3, h2.coo.column_indices);
    ASSERT_EQUAL_RANGES(h1.coo.values,         h1.coo.values         + 3, h2.coo.values);

    cusp::deallocate_matrix(h1);
    cusp::deallocate_matrix(h2);
    cusp::deallocate_matrix(d1);
    cusp::deallocate_matrix(d2);
}
DECLARE_UNITTEST(TestMemcpyHybMatrix);

