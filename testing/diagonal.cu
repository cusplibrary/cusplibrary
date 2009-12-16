#include <unittest/unittest.h>

#include <cusp/precond/diagonal.h>
#include <cusp/array2d.h>

template <class Space>
void TestCsrExpandRowOffsets(void)
{
    cusp::csr_matrix<int, float, Space> matrix(7, 1, 10);

    matrix.row_offsets[0] =  0;
    matrix.row_offsets[1] =  0;
    matrix.row_offsets[2] =  0;
    matrix.row_offsets[3] =  1;
    matrix.row_offsets[4] =  1;
    matrix.row_offsets[5] =  2;
    matrix.row_offsets[6] =  5;
    matrix.row_offsets[7] = 10;
    
    cusp::array1d<int, Space> expected(10);
    expected[0] = 2;
    expected[1] = 4;
    expected[2] = 5;
    expected[3] = 5;
    expected[4] = 5;
    expected[5] = 6;
    expected[6] = 6;
    expected[7] = 6;
    expected[8] = 6;
    expected[9] = 6;

    cusp::array1d<int, Space> output(10);
    cusp::precond::detail::expand_row_offsets(matrix, output);

    ASSERT_EQUAL(output, expected);
}
DECLARE_HOST_DEVICE_UNITTEST(TestCsrExpandRowOffsets);


template <class Space>
void TestCsrExtractDiagonal(void)
{
    {
        cusp::array2d<float, Space> A(2,2);
        A(0,0) = 1.0;  A(0,1) = 2.0;
        A(1,0) = 3.0;  A(1,1) = 4.0;

        cusp::array1d<float, Space> expected(2);
        expected[0] = 1.0;
        expected[1] = 4.0;

        cusp::array1d<float, Space> output;

        cusp::precond::detail::extract_diagonal(cusp::csr_matrix<int, float, Space>(A), output);

        ASSERT_EQUAL(output, expected);
    }
    
    {
        cusp::array2d<float, Space> A(2,2);
        A(0,0) = 0.0;  A(0,1) = 1.0;
        A(1,0) = 0.0;  A(1,1) = 2.0;

        cusp::array1d<float, Space> expected(2);
        expected[0] = 0.0;
        expected[1] = 2.0;

        cusp::array1d<float, Space> output;

        cusp::precond::detail::extract_diagonal(cusp::csr_matrix<int, float, Space>(A), output);

        ASSERT_EQUAL(output, expected);
    }
    
    {
        cusp::array2d<float, Space> A(5,5);
        A(0,0) = 0.0;  A(0,1) = 1.0;   A(0,2) = 2.0;   A(0,3) = 0.0;   A(0,4) = 0.0; 
        A(1,0) = 3.0;  A(1,1) = 4.0;   A(1,2) = 0.0;   A(1,3) = 0.0;   A(1,4) = 5.0;
        A(2,0) = 0.0;  A(2,1) = 0.0;   A(2,2) = 0.0;   A(2,3) = 0.0;   A(2,4) = 0.0;
        A(3,0) = 0.0;  A(3,1) = 6.0;   A(3,2) = 7.0;   A(3,3) = 8.0;   A(3,4) = 0.0;
        A(4,0) = 0.0;  A(4,1) = 0.0;   A(4,2) = 0.0;   A(4,3) = 0.0;   A(4,4) = 9.0;

        cusp::array1d<float, Space> expected(5);
        expected[0] = 0.0;
        expected[1] = 4.0;
        expected[2] = 0.0;
        expected[3] = 8.0;
        expected[4] = 9.0;

        cusp::array1d<float, Space> output;

        cusp::precond::detail::extract_diagonal(cusp::csr_matrix<int, float, Space>(A), output);

        ASSERT_EQUAL(output, expected);
    }

}
DECLARE_HOST_DEVICE_UNITTEST(TestCsrExtractDiagonal);


template <class Space>
void TestDiagonalPreconditioner(void)
{
    cusp::array2d<float, Space> A(5,5);
    A(0,0) = 1.0;  A(0,1) = 1.0;   A(0,2) = 2.0;   A(0,3) = 0.0;   A(0,4) = 0.0; 
    A(1,0) = 3.0;  A(1,1) = 2.0;   A(1,2) = 0.0;   A(1,3) = 0.0;   A(1,4) = 5.0;
    A(2,0) = 0.0;  A(2,1) = 0.0;   A(2,2) = 0.5;   A(2,3) = 0.0;   A(2,4) = 0.0;
    A(3,0) = 0.0;  A(3,1) = 6.0;   A(3,2) = 7.0;   A(3,3) = 4.0;   A(3,4) = 0.0;
    A(4,0) = 0.0;  A(4,1) = 8.0;   A(4,2) = 0.0;   A(4,3) = 0.0;   A(4,4) = 0.25;

    cusp::array1d<float, Space> input(5, 1.0);
    cusp::array1d<float, Space> expected(5);
    expected[0] = 1.00;
    expected[1] = 0.50;
    expected[2] = 2.00;
    expected[3] = 0.25;
    expected[4] = 4.00;

    cusp::array1d<float, Space> output(5, 0.0f);
   
    cusp::csr_matrix<int, float, Space> csr(A);
    cusp::precond::diagonal<float, Space> M(csr);

    M.multiply(input, output);

    ASSERT_EQUAL(output, expected);
}
DECLARE_HOST_DEVICE_UNITTEST(TestDiagonalPreconditioner);

