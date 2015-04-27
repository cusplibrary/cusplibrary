#include <unittest/unittest.h>
#include <cusp/array2d.h>
#include <cusp/gallery/poisson.h>
#include <cusp/lapack/lapack.h>

void TestGETRF(void)
{
    cusp::array2d<float, cusp::host_memory> A;
    cusp::array1d<int, cusp::host_memory> piv;

    cusp::gallery::poisson5pt(A, 4, 4);

    cusp::lapack::getrf(A, piv);

    cusp::array2d<float, cusp::host_memory> B(A.num_rows, 10);
    cusp::lapack::getrs(A, piv, B);
}
DECLARE_UNITTEST(TestGETRF);

void TestPOTRF(void)
{
    cusp::array2d<float, cusp::host_memory> A;

    cusp::gallery::poisson5pt(A, 4, 4);

    cusp::lapack::potrf(A);

    cusp::array2d<float, cusp::host_memory> B(A.num_rows, 10);
    cusp::lapack::potrs(A, B);
}
DECLARE_UNITTEST(TestPOTRF);

void TestSYTRF(void)
{
    cusp::array2d<float, cusp::host_memory> A;
    cusp::array1d<int, cusp::host_memory> piv;

    cusp::gallery::poisson5pt(A, 4, 4);

    cusp::lapack::sytrf(A, piv);

    cusp::array2d<float, cusp::host_memory> B(A.num_rows, 10);
    cusp::lapack::sytrs(A, piv, B);
}
DECLARE_UNITTEST(TestSYTRF);

void TestGESV(void)
{
    cusp::array2d<float, cusp::host_memory> A;
    cusp::array1d<int, cusp::host_memory> piv;

    cusp::gallery::poisson5pt(A, 4, 4);

    cusp::array2d<float, cusp::host_memory> B(A.num_rows, 10);
    cusp::lapack::gesv(A, B, piv);
}
DECLARE_UNITTEST(TestGESV);

void TestTRTRS(void)
{
    cusp::array2d<float, cusp::host_memory> A;

    cusp::gallery::poisson5pt(A, 4, 4);

    cusp::array2d<float, cusp::host_memory> B(A.num_rows, 10);
    cusp::lapack::trtrs(A, B);
}
DECLARE_UNITTEST(TestTRTRS);

void TestTRTRI(void)
{
    cusp::array2d<float, cusp::host_memory> A;

    cusp::gallery::poisson5pt(A, 4, 4);

    cusp::lapack::trtri(A);
}
DECLARE_UNITTEST(TestTRTRI);

void TestSYEV(void)
{
    cusp::array2d<float, cusp::host_memory> A;

    cusp::gallery::poisson5pt(A, 4, 4);

    cusp::array1d<float, cusp::host_memory> eigvals;
    cusp::array2d<float, cusp::host_memory> B(A.num_rows, A.num_cols);
    cusp::lapack::syev(A, eigvals, B);
}
DECLARE_UNITTEST(TestSYEV);

