#include <unittest/unittest.h>
#include <cusp/dia_matrix.h>

template <class Space>
void TestDiaMatrixBasicConstructor(void)
{
    cusp::dia_matrix<int, float, Space> matrix(4, 4, 7, 3, 4);

    ASSERT_EQUAL(matrix.num_rows,              4);
    ASSERT_EQUAL(matrix.num_cols,              4);
    ASSERT_EQUAL(matrix.num_entries,           7);
    ASSERT_EQUAL(matrix.num_diagonals,         3);
    ASSERT_EQUAL(matrix.stride,                4);
    ASSERT_EQUAL(matrix.diagonal_offsets.size(),  3);
    ASSERT_EQUAL(matrix.values.size(),           12);
}
DECLARE_HOST_DEVICE_UNITTEST(TestDiaMatrixBasicConstructor);

template <class Space>
void TestDiaMatrixCopyConstructor(void)
{
    cusp::dia_matrix<int, float, Space> matrix(4, 4, 7, 3, 4);

    matrix.diagonal_offsets[0] = -2;
    matrix.diagonal_offsets[1] =  0;
    matrix.diagonal_offsets[2] =  1;

    matrix.values[ 0] =  0; 
    matrix.values[ 1] =  0; 
    matrix.values[ 2] = 13; 
    matrix.values[ 3] = 16; 
    matrix.values[ 4] = 10; 
    matrix.values[ 5] =  0; 
    matrix.values[ 6] = 14; 
    matrix.values[ 7] =  0; 
    matrix.values[ 8] = 11; 
    matrix.values[ 9] = 12; 
    matrix.values[10] = 15; 
    matrix.values[11] =  0; 
    
    cusp::dia_matrix<int, float, Space> copy_of_matrix(matrix);

    ASSERT_EQUAL(copy_of_matrix.num_rows,              4);
    ASSERT_EQUAL(copy_of_matrix.num_cols,              4);
    ASSERT_EQUAL(copy_of_matrix.num_entries,           7);
    ASSERT_EQUAL(copy_of_matrix.num_diagonals,         3);
    ASSERT_EQUAL(copy_of_matrix.stride,                4);
    ASSERT_EQUAL(copy_of_matrix.diagonal_offsets.size(),  3);
    ASSERT_EQUAL(copy_of_matrix.values.size(),           12);

    ASSERT_EQUAL(copy_of_matrix.diagonal_offsets, matrix.diagonal_offsets);
    ASSERT_EQUAL(copy_of_matrix.values,           matrix.values);
}
DECLARE_HOST_DEVICE_UNITTEST(TestDiaMatrixCopyConstructor);

template <class Space>
void TestDiaMatrixResize(void)
{
    cusp::dia_matrix<int, float, Space> matrix;
    
    matrix.resize(4, 4, 7, 3, 4);

    ASSERT_EQUAL(matrix.num_rows,              4);
    ASSERT_EQUAL(matrix.num_cols,              4);
    ASSERT_EQUAL(matrix.num_entries,           7);
    ASSERT_EQUAL(matrix.num_diagonals,         3);
    ASSERT_EQUAL(matrix.stride,                4);
    ASSERT_EQUAL(matrix.diagonal_offsets.size(),  3);
    ASSERT_EQUAL(matrix.values.size(),           12);
}
DECLARE_HOST_DEVICE_UNITTEST(TestDiaMatrixResize);

template <class Space>
void TestDiaMatrixSwap(void)
{
    cusp::dia_matrix<int, float, Space> A(2, 2, 4, 3, 2);
    cusp::dia_matrix<int, float, Space> B(1, 3, 2, 2, 1);

    A.diagonal_offsets[0] = -1;
    A.diagonal_offsets[1] =  0;
    A.diagonal_offsets[2] =  1;

    A.values[0] = 10;
    A.values[1] =  0;
    A.values[2] = 20;
    A.values[3] = 30;
    A.values[4] = 40;
    A.values[5] = 40;

    B.diagonal_offsets[0] = 1;
    B.diagonal_offsets[1] = 2;

    B.values[0] = 10;
    B.values[1] = 20;
    
    cusp::dia_matrix<int, float, Space> A_copy(A);
    cusp::dia_matrix<int, float, Space> B_copy(B);

    A.swap(B);

    ASSERT_EQUAL(A.num_rows,         B_copy.num_rows);
    ASSERT_EQUAL(A.num_cols,         B_copy.num_cols);
    ASSERT_EQUAL(A.num_entries,      B_copy.num_entries);
    ASSERT_EQUAL(A.num_diagonals,    B_copy.num_diagonals);
    ASSERT_EQUAL(A.stride,           B_copy.stride);
    ASSERT_EQUAL(A.diagonal_offsets, B_copy.diagonal_offsets);
    ASSERT_EQUAL(A.values,           B_copy.values);
    
    ASSERT_EQUAL(B.num_rows,         A_copy.num_rows);
    ASSERT_EQUAL(B.num_cols,         A_copy.num_cols);
    ASSERT_EQUAL(B.num_entries,      A_copy.num_entries);
    ASSERT_EQUAL(B.num_diagonals,    A_copy.num_diagonals);
    ASSERT_EQUAL(B.stride,           A_copy.stride);
    ASSERT_EQUAL(B.diagonal_offsets, A_copy.diagonal_offsets);
    ASSERT_EQUAL(B.values,           A_copy.values);
}
DECLARE_HOST_DEVICE_UNITTEST(TestDiaMatrixSwap);

