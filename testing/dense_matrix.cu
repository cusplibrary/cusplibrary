#include <unittest/unittest.h>
#include <cusp/dense_matrix.h>

template <class Space>
void TestDenseMatrixBasicConstructor(void)
{
    cusp::dense_matrix<float, Space> matrix(3, 2);

    ASSERT_EQUAL(matrix.num_rows,       3);
    ASSERT_EQUAL(matrix.num_cols,       2);
    ASSERT_EQUAL(matrix.num_entries,    6);
    ASSERT_EQUAL(matrix.values.size(),  6);
}
DECLARE_HOST_DEVICE_UNITTEST(TestDenseMatrixBasicConstructor);

template <class Space>
void TestDenseMatrixCopyConstructor(void)
{
    cusp::dense_matrix<float, Space> matrix(3, 2);

    matrix.values[0] = 0; 
    matrix.values[1] = 1; 
    matrix.values[2] = 2; 
    matrix.values[3] = 3; 
    matrix.values[4] = 4; 
    matrix.values[5] = 5; 
    
    cusp::dense_matrix<float, Space> copy_of_matrix(matrix);
    
    ASSERT_EQUAL(matrix.num_rows,       3);
    ASSERT_EQUAL(matrix.num_cols,       2);
    ASSERT_EQUAL(matrix.num_entries,    6);
    ASSERT_EQUAL(matrix.values.size(),  6);

    ASSERT_EQUAL(copy_of_matrix.values, matrix.values);
}
DECLARE_HOST_DEVICE_UNITTEST(TestDenseMatrixCopyConstructor);

template <class Space>
void TestDenseMatrixRowMajor(void)
{
    cusp::dense_matrix<float, Space, cusp::row_major> matrix(2,3);

    matrix(0,0) = 10;  matrix(0,1) = 20;  matrix(0,2) = 30; 
    matrix(1,0) = 40;  matrix(1,1) = 50;  matrix(1,2) = 60;

    ASSERT_EQUAL(matrix(0,0), 10);
    ASSERT_EQUAL(matrix(0,1), 20);
    ASSERT_EQUAL(matrix(0,2), 30);
    ASSERT_EQUAL(matrix(1,0), 40);
    ASSERT_EQUAL(matrix(1,1), 50);
    ASSERT_EQUAL(matrix(1,2), 60);

    ASSERT_EQUAL(matrix.values[0], 10);
    ASSERT_EQUAL(matrix.values[1], 20);
    ASSERT_EQUAL(matrix.values[2], 30);
    ASSERT_EQUAL(matrix.values[3], 40);
    ASSERT_EQUAL(matrix.values[4], 50);
    ASSERT_EQUAL(matrix.values[5], 60);
}
DECLARE_HOST_DEVICE_UNITTEST(TestDenseMatrixRowMajor);

template <class Space>
void TestDenseMatrixColumnMajor(void)
{
    cusp::dense_matrix<float, Space, cusp::column_major> matrix(2,3);
    
    matrix(0,0) = 10;  matrix(0,1) = 20;  matrix(0,2) = 30; 
    matrix(1,0) = 40;  matrix(1,1) = 50;  matrix(1,2) = 60;
    
    ASSERT_EQUAL(matrix(0,0), 10);
    ASSERT_EQUAL(matrix(0,1), 20);
    ASSERT_EQUAL(matrix(0,2), 30);
    ASSERT_EQUAL(matrix(1,0), 40);
    ASSERT_EQUAL(matrix(1,1), 50);
    ASSERT_EQUAL(matrix(1,2), 60);

    ASSERT_EQUAL(matrix.values[0], 10);
    ASSERT_EQUAL(matrix.values[1], 40);
    ASSERT_EQUAL(matrix.values[2], 20);
    ASSERT_EQUAL(matrix.values[3], 50);
    ASSERT_EQUAL(matrix.values[4], 30);
    ASSERT_EQUAL(matrix.values[5], 60);
}
DECLARE_HOST_DEVICE_UNITTEST(TestDenseMatrixColumnMajor);

template <class Space>
void TestDenseMatrixResize(void)
{
    cusp::dense_matrix<float, Space> matrix;
    
    matrix.resize(3, 2);

    ASSERT_EQUAL(matrix.num_rows,       3);
    ASSERT_EQUAL(matrix.num_cols,       2);
    ASSERT_EQUAL(matrix.num_entries,    6);
    ASSERT_EQUAL(matrix.values.size(),  6);
}
DECLARE_HOST_DEVICE_UNITTEST(TestDenseMatrixResize);

template <class Space>
void TestDenseMatrixSwap(void)
{
    cusp::dense_matrix<float, Space> A(2,2);
    cusp::dense_matrix<float, Space> B(3,1);
    
    A(0,0) = 10;  A(0,1) = 20;
    A(1,0) = 30;  A(1,1) = 40;
    
    B(0,0) = 50;
    B(1,0) = 60;
    B(2,0) = 70;

    cusp::dense_matrix<float, Space> A_copy(A);
    cusp::dense_matrix<float, Space> B_copy(B);

    A.swap(B);

    ASSERT_EQUAL(A.num_rows,    B_copy.num_rows);
    ASSERT_EQUAL(A.num_cols,    B_copy.num_cols);
    ASSERT_EQUAL(A.num_entries, B_copy.num_entries);
    ASSERT_EQUAL(A.values,      B_copy.values);
    
    ASSERT_EQUAL(B.num_rows,    A_copy.num_rows);
    ASSERT_EQUAL(B.num_cols,    A_copy.num_cols);
    ASSERT_EQUAL(B.num_entries, A_copy.num_entries);
    ASSERT_EQUAL(B.values,      A_copy.values);
}
DECLARE_HOST_DEVICE_UNITTEST(TestDenseMatrixSwap);

void TestDenseMatrixRebind(void)
{
    typedef cusp::dense_matrix<float, cusp::host>  HostMatrix;
    typedef HostMatrix::rebind<cusp::device>::type DeviceMatrix;

    HostMatrix   h_matrix(10,10);
    DeviceMatrix d_matrix(h_matrix);

    ASSERT_EQUAL(h_matrix.num_entries, d_matrix.num_entries);
}
DECLARE_UNITTEST(TestDenseMatrixRebind);

