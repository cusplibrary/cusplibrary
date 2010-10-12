#include <unittest/unittest.h>
#include <cusp/array2d.h>

template <class Space>
void TestArray2dBasicConstructor(void)
{
    cusp::array2d<float, Space> matrix(3, 2);

    ASSERT_EQUAL(matrix.num_rows,       3);
    ASSERT_EQUAL(matrix.num_cols,       2);
    ASSERT_EQUAL(matrix.num_entries,    6);
    ASSERT_EQUAL(matrix.values.size(),  6);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dBasicConstructor);

template <class Space>
void TestArray2dFillConstructor(void)
{
    cusp::array2d<float, Space> matrix(3, 2, 13.0f);

    ASSERT_EQUAL(matrix.num_rows,       3);
    ASSERT_EQUAL(matrix.num_cols,       2);
    ASSERT_EQUAL(matrix.num_entries,    6);
    ASSERT_EQUAL(matrix.values.size(),  6);
    ASSERT_EQUAL(matrix.values[0], 13.0f);
    ASSERT_EQUAL(matrix.values[1], 13.0f);
    ASSERT_EQUAL(matrix.values[2], 13.0f);
    ASSERT_EQUAL(matrix.values[3], 13.0f);
    ASSERT_EQUAL(matrix.values[4], 13.0f);
    ASSERT_EQUAL(matrix.values[5], 13.0f);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dFillConstructor);

template <class Space>
void TestArray2dCopyConstructor(void)
{
    cusp::array2d<float, Space> matrix(3, 2);

    matrix.values[0] = 0; 
    matrix.values[1] = 1; 
    matrix.values[2] = 2; 
    matrix.values[3] = 3; 
    matrix.values[4] = 4; 
    matrix.values[5] = 5; 
    
    cusp::array2d<float, Space> copy_of_matrix(matrix);
    
    ASSERT_EQUAL(matrix.num_rows,       3);
    ASSERT_EQUAL(matrix.num_cols,       2);
    ASSERT_EQUAL(matrix.num_entries,    6);
    ASSERT_EQUAL(matrix.values.size(),  6);

    ASSERT_EQUAL(copy_of_matrix.values, matrix.values);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dCopyConstructor);

template <class Space>
void TestArray2dRowMajor(void)
{
    cusp::array2d<float, Space, cusp::row_major> matrix(2,3);

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
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dRowMajor);

template <class Space>
void TestArray2dColumnMajor(void)
{
    cusp::array2d<float, Space, cusp::column_major> matrix(2,3);
    
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
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dColumnMajor);

template <class Space>
void TestArray2dMixedOrientations(void)
{
    cusp::array2d<float, Space, cusp::row_major>    R(2,3);
    cusp::array2d<float, Space, cusp::column_major> C(2,3);

    R(0,0) = 10;  R(0,1) = 20;  R(0,2) = 30; 
    R(1,0) = 40;  R(1,1) = 50;  R(1,2) = 60;

    C = R;
    ASSERT_EQUAL(C(0,0), 10);
    ASSERT_EQUAL(C(0,1), 20);
    ASSERT_EQUAL(C(0,2), 30);
    ASSERT_EQUAL(C(1,0), 40);
    ASSERT_EQUAL(C(1,1), 50);
    ASSERT_EQUAL(C(1,2), 60);

    R = C;
    ASSERT_EQUAL(R(0,0), 10);
    ASSERT_EQUAL(R(0,1), 20);
    ASSERT_EQUAL(R(0,2), 30);
    ASSERT_EQUAL(R(1,0), 40);
    ASSERT_EQUAL(R(1,1), 50);
    ASSERT_EQUAL(R(1,2), 60);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dMixedOrientations);

template <class Space>
void TestArray2dResize(void)
{
    cusp::array2d<float, Space> matrix;
    
    matrix.resize(3, 2);

    ASSERT_EQUAL(matrix.num_rows,       3);
    ASSERT_EQUAL(matrix.num_cols,       2);
    ASSERT_EQUAL(matrix.num_entries,    6);
    ASSERT_EQUAL(matrix.values.size(),  6);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dResize);

template <class Space>
void TestArray2dSwap(void)
{
    cusp::array2d<float, Space> A(2,2);
    cusp::array2d<float, Space> B(3,1);
    
    A(0,0) = 10;  A(0,1) = 20;
    A(1,0) = 30;  A(1,1) = 40;
    
    B(0,0) = 50;
    B(1,0) = 60;
    B(2,0) = 70;

    cusp::array2d<float, Space> A_copy(A);
    cusp::array2d<float, Space> B_copy(B);

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
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dSwap);

void TestArray2dRebind(void)
{
    typedef cusp::array2d<float, cusp::host_memory>       HostMatrix;
    typedef HostMatrix::rebind<cusp::device_memory>::type DeviceMatrix;

    HostMatrix   h_matrix(10,10);
    DeviceMatrix d_matrix(h_matrix);

    ASSERT_EQUAL(h_matrix.num_entries, d_matrix.num_entries);
}
DECLARE_UNITTEST(TestArray2dRebind);

template <typename MemorySpace>
void TestArray2dEquality(void)
{
    cusp::array2d<float, cusp::host_memory, cusp::row_major>    A(2,2);
    cusp::array2d<float, cusp::host_memory, cusp::row_major>    B(2,3);
    cusp::array2d<float, cusp::host_memory, cusp::column_major> C(2,3);
    cusp::array2d<float, cusp::host_memory, cusp::column_major> D(2,2);

    A(0,0) = 1;  A(0,1) = 2;
    A(1,0) = 4;  A(1,1) = 5;
    
    B(0,0) = 1;  B(0,1) = 2;  B(0,2) = 3;
    B(1,0) = 7;  B(1,1) = 5;  B(1,2) = 6;
    
    C(0,0) = 1;  C(0,1) = 2;  C(0,2) = 3;
    C(1,0) = 4;  C(1,1) = 5;  C(1,2) = 6;
    
    D(0,0) = 1;  D(0,1) = 2;
    D(1,0) = 4;  D(1,1) = 5;

    ASSERT_EQUAL(A == A, true);
    ASSERT_EQUAL(B == B, true);
    ASSERT_EQUAL(C == C, true);
    ASSERT_EQUAL(D == D, true);
    
    ASSERT_EQUAL(A != A, false);
    ASSERT_EQUAL(B != B, false);
    ASSERT_EQUAL(C != C, false);
    ASSERT_EQUAL(D != D, false);

    ASSERT_EQUAL(A == B, false);
//    ASSERT_EQUAL(A == C,  true);
//    ASSERT_EQUAL(A == D, false);
//    ASSERT_EQUAL(B == C, false);
//    ASSERT_EQUAL(B == D, false);
    ASSERT_EQUAL(C == D, false);
    
    ASSERT_EQUAL(A != B,  true);
//    ASSERT_EQUAL(A != C, false);
//    ASSERT_EQUAL(A != D,  true);
//    ASSERT_EQUAL(B != C,  true);
//    ASSERT_EQUAL(B != D,  true);
    ASSERT_EQUAL(C != D,  true);
}
DECLARE_HOST_DEVICE_UNITTEST(TestArray2dEquality);

