#include <unittest/unittest.h>
#include <cusp/csr_matrix.h>

template <class Space>
void TestCsrPatternBasicConstructor(void)
{
    cusp::csr_pattern<int, Space> pattern(3, 2, 6);

    ASSERT_EQUAL(pattern.num_rows,              3);
    ASSERT_EQUAL(pattern.num_cols,              2);
    ASSERT_EQUAL(pattern.num_entries,           6);
    ASSERT_EQUAL(pattern.row_offsets.size(),    4);
    ASSERT_EQUAL(pattern.column_indices.size(), 6);
}
DECLARE_HOST_DEVICE_UNITTEST(TestCsrPatternBasicConstructor);

template <class Space>
void TestCsrPatternCopyConstructor(void)
{
    cusp::csr_pattern<int, Space> pattern(3, 2, 6);

    pattern.row_offsets[0] = 0;
    pattern.row_offsets[1] = 2;
    pattern.row_offsets[2] = 4;
    pattern.row_offsets[3] = 6;
    
    pattern.column_indices[0] = 0;
    pattern.column_indices[1] = 1;
    pattern.column_indices[2] = 0;
    pattern.column_indices[3] = 1;
    pattern.column_indices[4] = 0;
    pattern.column_indices[5] = 1;

    cusp::csr_pattern<int, Space> copy_of_pattern(pattern);
    
    ASSERT_EQUAL(copy_of_pattern.num_rows,              3);
    ASSERT_EQUAL(copy_of_pattern.num_cols,              2);
    ASSERT_EQUAL(copy_of_pattern.num_entries,           6);
    ASSERT_EQUAL(copy_of_pattern.row_offsets.size(),    4);
    ASSERT_EQUAL(copy_of_pattern.column_indices.size(), 6);
   
    ASSERT_EQUAL(copy_of_pattern.row_offsets,    pattern.row_offsets);
    ASSERT_EQUAL(copy_of_pattern.column_indices, pattern.column_indices);
}
DECLARE_HOST_DEVICE_UNITTEST(TestCsrPatternCopyConstructor);

template <class Space>
void TestCsrMatrixBasicConstructor(void)
{
    cusp::csr_matrix<int, float, Space> matrix(3, 2, 6);

    ASSERT_EQUAL(matrix.num_rows,              3);
    ASSERT_EQUAL(matrix.num_cols,              2);
    ASSERT_EQUAL(matrix.num_entries,           6);
    ASSERT_EQUAL(matrix.row_offsets.size(),    4);
    ASSERT_EQUAL(matrix.column_indices.size(), 6);
    ASSERT_EQUAL(matrix.values.size(),         6);
}
DECLARE_HOST_DEVICE_UNITTEST(TestCsrMatrixBasicConstructor);

template <class Space>
void TestCsrMatrixCopyConstructor(void)
{
    cusp::csr_matrix<int, float, Space> matrix(3, 2, 6);

    matrix.row_offsets[0] = 0;
    matrix.row_offsets[1] = 2;
    matrix.row_offsets[2] = 4;
    matrix.row_offsets[3] = 6;
    
    matrix.column_indices[0] = 0;  matrix.values[0] = 0; 
    matrix.column_indices[1] = 1;  matrix.values[1] = 1;
    matrix.column_indices[2] = 0;  matrix.values[2] = 2;
    matrix.column_indices[3] = 1;  matrix.values[3] = 3;
    matrix.column_indices[4] = 0;  matrix.values[4] = 4;
    matrix.column_indices[5] = 1;  matrix.values[5] = 5;

    cusp::csr_matrix<int, float, Space> copy_of_matrix(matrix);
    
    ASSERT_EQUAL(copy_of_matrix.num_rows,              3);
    ASSERT_EQUAL(copy_of_matrix.num_cols,              2);
    ASSERT_EQUAL(copy_of_matrix.num_entries,           6);
    ASSERT_EQUAL(copy_of_matrix.row_offsets.size(),    4);
    ASSERT_EQUAL(copy_of_matrix.column_indices.size(), 6);
    ASSERT_EQUAL(copy_of_matrix.values.size(),         6);
   
    ASSERT_EQUAL(copy_of_matrix.row_offsets,    matrix.row_offsets);
    ASSERT_EQUAL(copy_of_matrix.column_indices, matrix.column_indices);
    ASSERT_EQUAL(copy_of_matrix.values,         matrix.values);
}
DECLARE_HOST_DEVICE_UNITTEST(TestCsrMatrixCopyConstructor);

template <class Space>
void TestCsrMatrixResize(void)
{
    cusp::csr_matrix<int, float, Space> matrix;
    
    matrix.resize(3, 2, 6);

    ASSERT_EQUAL(matrix.num_rows,              3);
    ASSERT_EQUAL(matrix.num_cols,              2);
    ASSERT_EQUAL(matrix.num_entries,           6);
    ASSERT_EQUAL(matrix.row_offsets.size(),    4);
    ASSERT_EQUAL(matrix.column_indices.size(), 6);
    ASSERT_EQUAL(matrix.values.size(),         6);
}
DECLARE_HOST_DEVICE_UNITTEST(TestCsrMatrixResize);

template <class Space>
void TestCsrMatrixSwap(void)
{
    cusp::csr_matrix<int, float, Space> A(1, 2, 2);
    cusp::csr_matrix<int, float, Space> B(3, 1, 3);
  
    A.row_offsets[0] = 0;
    A.row_offsets[1] = 2;
    
    A.column_indices[0] = 0;  A.values[0] = 0;
    A.column_indices[1] = 1;  A.values[1] = 1;
    
    B.row_offsets[0] = 0;
    B.row_offsets[1] = 1;
    B.row_offsets[2] = 2;
    B.row_offsets[3] = 3;
    
    B.column_indices[0] = 0;  B.values[0] = 0;
    B.column_indices[1] = 0;  B.values[1] = 1;
    B.column_indices[2] = 0;  B.values[2] = 2;
    
    cusp::csr_matrix<int, float, Space> A_copy(A);
    cusp::csr_matrix<int, float, Space> B_copy(B);

    A.swap(B);

    ASSERT_EQUAL(A.num_rows,              3);
    ASSERT_EQUAL(A.num_cols,              1);
    ASSERT_EQUAL(A.num_entries,           3);
    ASSERT_EQUAL(A.row_offsets,    B_copy.row_offsets);
    ASSERT_EQUAL(A.column_indices, B_copy.column_indices);
    ASSERT_EQUAL(A.values,         B_copy.values);

    ASSERT_EQUAL(B.num_rows,              1);
    ASSERT_EQUAL(B.num_cols,              2);
    ASSERT_EQUAL(B.num_entries,           2);
    ASSERT_EQUAL(B.row_offsets,    A_copy.row_offsets);
    ASSERT_EQUAL(B.column_indices, A_copy.column_indices);
    ASSERT_EQUAL(B.values,         A_copy.values);
}
DECLARE_HOST_DEVICE_UNITTEST(TestCsrMatrixSwap);

void TestCsrPatternRebind(void)
{
    typedef cusp::csr_pattern<int, cusp::host>      HostPattern;
    typedef HostPattern::rebind<cusp::device>::type DevicePattern;

    HostPattern   h_pattern(10,10,100);
    DevicePattern d_pattern(h_pattern);

    ASSERT_EQUAL(h_pattern.num_entries, d_pattern.num_entries);
}
DECLARE_UNITTEST(TestCsrPatternRebind);

void TestCsrMatrixRebind(void)
{
    typedef cusp::csr_matrix<int, float, cusp::host> HostMatrix;
    typedef HostMatrix::rebind<cusp::device>::type   DeviceMatrix;

    HostMatrix   h_matrix(10,10,100);
    DeviceMatrix d_matrix(h_matrix);

    ASSERT_EQUAL(h_matrix.num_entries, d_matrix.num_entries);
}
DECLARE_UNITTEST(TestCsrMatrixRebind);

