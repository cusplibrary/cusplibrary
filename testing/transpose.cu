#include <unittest/unittest.h>

#include <cusp/transpose.h>

#include <cusp/array2d.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

template <typename MatrixType>
void initialize_matrix(MatrixType& matrix)
{
    cusp::array2d<float, cusp::host_memory> D(4,3);
    
    D(0,0) = 10.25;  D(0,1) = 11.00;  D(0,2) =  0.00; 
    D(1,0) =  0.00;  D(1,1) =  0.00;  D(1,2) = 12.50; 
    D(2,0) = 13.75;  D(2,1) =  0.00;  D(2,2) = 14.00; 
    D(3,0) =  0.00;  D(3,1) = 16.50;  D(3,2) =  0.00; 

    matrix = D;
}

template <typename MatrixType>
void verify_result(const MatrixType& matrix)
{
    typedef typename MatrixType::value_type ValueType;
    
    ASSERT_EQUAL(matrix.num_rows,    3);
    ASSERT_EQUAL(matrix.num_cols,    4);

    cusp::array2d<ValueType, cusp::host_memory> dense(matrix);
   
    ASSERT_EQUAL(dense(0,0), 10.25);  
    ASSERT_EQUAL(dense(0,1),  0.00);  
    ASSERT_EQUAL(dense(0,2), 13.75);  
    ASSERT_EQUAL(dense(0,3),  0.00);
    ASSERT_EQUAL(dense(1,0), 11.00);
    ASSERT_EQUAL(dense(1,1),  0.00);
    ASSERT_EQUAL(dense(1,2),  0.00);
    ASSERT_EQUAL(dense(1,3), 16.50);
    ASSERT_EQUAL(dense(2,0),  0.00);
    ASSERT_EQUAL(dense(2,1), 12.50);
    ASSERT_EQUAL(dense(2,2), 14.00);
    ASSERT_EQUAL(dense(2,3),  0.00);
}


template <class MatrixType>
void TestTranspose(MatrixType mtx)
{
    MatrixType A, At;

    initialize_matrix(A);
    
    cusp::transpose(A, At);

    verify_result(At);
}



///////////////////////
// Instantiate Tests //
///////////////////////
template <class Space>
void TestTransposeArray2d(void)
{
    TestTranspose(cusp::array2d<float, Space, cusp::row_major>());
    TestTranspose(cusp::array2d<float, Space, cusp::column_major>());
}
DECLARE_HOST_DEVICE_UNITTEST(TestTransposeArray2d);

template <class Space>
void TestTransposeCooMatrix(void)
{
    TestTranspose(cusp::coo_matrix<int, float, Space>());
}
DECLARE_HOST_DEVICE_UNITTEST(TestTransposeCooMatrix);

template <class Space>
void TestTransposeCsrMatrix(void)
{
    TestTranspose(cusp::csr_matrix<int, float, Space>());
}
DECLARE_HOST_DEVICE_UNITTEST(TestTransposeCsrMatrix);

template <class Space>
void TestTransposeDiaMatrix(void)
{
    TestTranspose(cusp::dia_matrix<int, float, Space>());
}
DECLARE_HOST_DEVICE_UNITTEST(TestTransposeDiaMatrix);

template <class Space>
void TestTransposeEllMatrix(void)
{
    TestTranspose(cusp::ell_matrix<int, float, Space>());
}
DECLARE_HOST_DEVICE_UNITTEST(TestTransposeEllMatrix);

template <class Space>
void TestTransposeHybMatrix(void)
{
    TestTranspose(cusp::hyb_matrix<int, float, Space>());
}
DECLARE_HOST_DEVICE_UNITTEST(TestTransposeHybMatrix);

