#include <unittest/unittest.h>

#include <cusp/precond/diagonal.h>
#include <cusp/array2d.h>

template <class MatrixType>
void TestDiagonalPreconditioner(void)
{
    typedef typename MatrixType::memory_space Space;

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

    MatrixType M(A);
    cusp::precond::diagonal<float, Space> D(M);

    D.multiply(input, output);

    ASSERT_EQUAL(output, expected);
}

template <class Space>
void TestDiagonalPreconditionerCooMarix(void)
{
    TestDiagonalPreconditioner< cusp::coo_matrix<int,float,Space> >();
}
DECLARE_HOST_DEVICE_UNITTEST(TestDiagonalPreconditionerCooMarix);

template <class Space>
void TestDiagonalPreconditionerCsrMarix(void)
{
    TestDiagonalPreconditioner< cusp::csr_matrix<int,float,Space> >();
}
DECLARE_HOST_DEVICE_UNITTEST(TestDiagonalPreconditionerCsrMarix);

template <class Space>
void TestDiagonalPreconditionerEllMarix(void)
{
    TestDiagonalPreconditioner< cusp::ell_matrix<int,float,Space> >();
}
DECLARE_HOST_DEVICE_UNITTEST(TestDiagonalPreconditionerEllMarix);

template <class Space>
void TestDiagonalPreconditionerHybMarix(void)
{
    TestDiagonalPreconditioner< cusp::hyb_matrix<int,float,Space> >();
}
DECLARE_HOST_DEVICE_UNITTEST(TestDiagonalPreconditionerHybMarix);

