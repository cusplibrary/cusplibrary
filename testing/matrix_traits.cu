#include <unittest/unittest.h>

#include <cusp/detail/matrix_traits.h>

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

typedef cusp::array1d<float, cusp::host_memory> A1D;
typedef cusp::array2d<float, cusp::host_memory> A2D;
typedef cusp::coo_matrix<int, float, cusp::host_memory> COO;
typedef cusp::csr_matrix<int, float, cusp::host_memory> CSR;
typedef cusp::dia_matrix<int, float, cusp::host_memory> DIA;
typedef cusp::ell_matrix<int, float, cusp::host_memory> ELL;
typedef cusp::hyb_matrix<int, float, cusp::host_memory> HYB;

void TestMatrixFormatArray1d(void)
{
    typedef cusp::detail::matrix_format<A1D>::type format;
    ASSERT_EQUAL((bool) (thrust::detail::is_same<format,cusp::detail::array1d_format_tag>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::detail::sparse_format_tag>::value), false);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::detail::dense_format_tag>::value), true);
}
DECLARE_UNITTEST(TestMatrixFormatArray1d);

void TestMatrixFormatArray2d(void)
{
    typedef cusp::detail::matrix_format<A2D>::type format;
    ASSERT_EQUAL((bool) (thrust::detail::is_same<format,cusp::detail::array2d_format_tag>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::detail::sparse_format_tag>::value),false);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::detail::dense_format_tag>::value), true);
}
DECLARE_UNITTEST(TestMatrixFormatArray2d);

void TestMatrixFormatCooMatrix(void)
{
    typedef cusp::detail::matrix_format<COO>::type format;
    ASSERT_EQUAL((bool) (thrust::detail::is_same<format,cusp::detail::coo_format_tag>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::detail::sparse_format_tag>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::detail::dense_format_tag>::value), false);
}
DECLARE_UNITTEST(TestMatrixFormatCooMatrix);

void TestMatrixFormatCsrMatrix(void)
{
    typedef cusp::detail::matrix_format<CSR>::type format;
    ASSERT_EQUAL((bool) (thrust::detail::is_same<format,cusp::detail::csr_format_tag>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::detail::sparse_format_tag>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::detail::dense_format_tag>::value), false);
}
DECLARE_UNITTEST(TestMatrixFormatCsrMatrix);

void TestMatrixFormatDiaMatrix(void)
{
    typedef cusp::detail::matrix_format<DIA>::type format;
    ASSERT_EQUAL((bool) (thrust::detail::is_same<format,cusp::detail::dia_format_tag>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::detail::sparse_format_tag>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::detail::dense_format_tag>::value), false);
}
DECLARE_UNITTEST(TestMatrixFormatDiaMatrix);

void TestMatrixFormatEllMatrix(void)
{
    typedef cusp::detail::matrix_format<ELL>::type format;
    ASSERT_EQUAL((bool) (thrust::detail::is_same<format,cusp::detail::ell_format_tag>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::detail::sparse_format_tag>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::detail::dense_format_tag>::value), false);
}
DECLARE_UNITTEST(TestMatrixFormatEllMatrix);

void TestMatrixFormatHybMatrix(void)
{
    typedef cusp::detail::matrix_format<HYB>::type format;
    ASSERT_EQUAL((bool) (thrust::detail::is_same<format,cusp::detail::hyb_format_tag>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::detail::sparse_format_tag>::value), true);
    ASSERT_EQUAL((bool) (thrust::detail::is_convertible<format,cusp::detail::dense_format_tag>::value), false);
}
DECLARE_UNITTEST(TestMatrixFormatHybMatrix);

