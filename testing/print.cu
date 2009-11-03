#include <unittest/unittest.h>

#include <cusp/io.h>
#include <cusp/print.h>

void TestPrintMatrix(void)
{
    cusp::coo_matrix<int, float, cusp::host_memory> coo;
    cusp::array2d<float, cusp::host_memory> arr;

    // load matrix
    cusp::read_matrix_market_file(coo, "data/test/coordinate_real_general.mtx");

    // convert to array2d
    arr = coo;

    //cusp::print_matrix(coo);
    //cusp::print_matrix(dense);
}
DECLARE_UNITTEST(TestPrintMatrix);

