#include <cusp/csr_matrix.h>
#include <cusp/print.h>

#include <cusp/gallery/grid.h>
#include <cusp/io/matrix_market.h>

#include "../timer.h"
#include <cusp/graph/maximum_flow.h>

int main(int argc, char*argv[])
{
    srand(time(NULL));

    typedef int   IndexType;
    typedef int ValueType;
    typedef cusp::device_memory MemorySpace;

    size_t size = 1024;

    /*cusp::array2d<ValueType,cusp::host_memory> dense(6,6,ValueType(0));
    dense(0,1) = 16; dense(0,2) = 13;
    dense(1,2) = 10; dense(1,3) = 12;
    dense(2,1) = 4;  dense(2,4) = 14;
    dense(3,2) = 9;  dense(3,5) = 20;
    dense(4,3) = 7;  dense(4,5) = 4;

    cusp::csr_matrix<IndexType, ValueType, MemorySpace> A(dense);*/
    cusp::csr_matrix<IndexType, ValueType, MemorySpace> A;

    if (argc == 1)
    {
        // no input file was specified, generate an example
        std::cout << "Generated matrix (grid2d) ";
        cusp::gallery::grid2d(A, size, size);
    }
    else if (argc == 2)
    {
        // an input file was specified, read it from disk
        cusp::io::read_matrix_market_file(A, argv[1]);
        std::cout << "Read matrix (" << argv[1] << ") ";
    }

    std::cout << "with shape ("  << A.num_rows << "," << A.num_cols << ") and "
              << A.num_entries << " entries" << "\n\n";


    srand (time(NULL));
    cusp::array1d<ValueType,cusp::host_memory> values(A.num_entries);
    for( int index = 0; index < A.num_entries; index++ )
        values[index] = (rand() % 100) + 4;
    A.values = values;

    cusp::csr_matrix<IndexType, ValueType, cusp::host_memory> A_h(A);

    IndexType src = 0;
    IndexType sink = A.num_rows - 1;

    timer flow_time;
    std::cout << "Device Max-flow : " << cusp::graph::maximum_flow(A, src, sink) << std::endl;;
    printf("Device Max-flow time : %4.2f (ms)\n", flow_time.milliseconds_elapsed());

    cusp::graph::detail::host::PushRelabel host_push_relabel(A_h);
    timer host_flow_time;
    std::cout << "Host Max-flow : " << host_push_relabel.GetMaxFlow(src, sink) << std::endl;;
    printf("Host Max-flow time : %4.2f (ms)\n", host_flow_time.milliseconds_elapsed());

    return EXIT_SUCCESS;
}

