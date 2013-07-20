#include <cusp/csr_matrix.h>
#include <cusp/print.h>

#include <cusp/gallery/grid.h>
#include <cusp/graph/maximum_flow.h>
#include <cusp/io/matrix_market.h>

#include "../timer.h"

template<typename MemorySpace, typename MatrixType>
void MaxFlow(const MatrixType& G)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;
    typedef cusp::csr_matrix<IndexType,ValueType,MemorySpace> GraphType;

    GraphType G_flow(G);
    cusp::array1d<IndexType,MemorySpace> min_cut_edges(G.num_entries);
    cusp::array1d<ValueType,MemorySpace> flow(G.num_entries);

    IndexType source = 0;
    IndexType sink = G.num_rows - 1;

    {
        timer t;
        ValueType capacity = cusp::graph::maximum_flow(G_flow, flow, source, sink);
        std::cout << "Max-Flow time : " << t.milliseconds_elapsed() << " (ms), with capacity : " << capacity << std::endl;
    }

    {
        timer t;
        size_t min_cut_size = cusp::graph::max_flow_to_min_cut(G_flow, flow, source, min_cut_edges);
        std::cout << "\tMin-Cut time : " << t.milliseconds_elapsed() << " (ms), found " << min_cut_size << " edge cuts." << std::endl;
    }
}

int main(int argc, char*argv[])
{
    srand(time(NULL));

    typedef int IndexType;
    typedef int ValueType;
    typedef cusp::host_memory MemorySpace;

    cusp::csr_matrix<IndexType, ValueType, MemorySpace> A;
    size_t size = 4;

    if (argc == 1)
    {
        // no input file was specified, generate an example
        std::cout << "Generated matrix (2D Grid) ";
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

    // Generate random capacities in range [4,100)
    for( size_t index = 0; index < A.num_entries; index++ )
        A.values[index] = (rand() % 100) + 4;

    std::cout << " Device ";
    MaxFlow<cusp::device_memory>(A);

    std::cout << " Host ";
    MaxFlow<cusp::host_memory>(A);

    std::cout << std::endl;

    return EXIT_SUCCESS;
}

