#include <unittest/unittest.h>

#include <cusp/io.h>
#include <cusp/print.h>
#include <cusp/convert.h>

void TestPrintMatrix(void)
{
    cusp::coo_matrix<int, float, cusp::host_memory> coo;
    cusp::dense_matrix<float, cusp::host_memory> dense;

    // load matrix
    cusp::load_matrix_market_file(coo, "data/test/coordinate_real_general.mtx");

    // convert to dense_matrix
    cusp::convert_matrix(dense, coo);

//    cusp::print_matrix(coo);
//    cusp::print_matrix(dense);

    cusp::deallocate_matrix(coo);
    cusp::deallocate_matrix(dense);
}
DECLARE_UNITTEST(TestPrintMatrix);

