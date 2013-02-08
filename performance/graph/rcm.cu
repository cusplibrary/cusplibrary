#include <cusp/csr_matrix.h>

#include <cusp/gallery/poisson.h>
#include <cusp/graph/detail/rcm.inl>
#include <cusp/io/matrix_market.h>

#include "../timer.h"

template<typename MemorySpace, typename MatrixType>
void RCM(const MatrixType& G)
{
    typedef typename MatrixType::index_type IndexType;
    typedef cusp::csr_matrix<IndexType,IndexType,MemorySpace> GraphType;
    typedef cusp::array1d<IndexType,MemorySpace> Array;

    GraphType G_bfs(G);
    timer t;
    cusp::graph::rcm(G_bfs);
    std::cout << " RCM time : " << t.milliseconds_elapsed() << " (ms)." << std::endl;
}

int main(int argc, char*argv[])
{
    srand(time(NULL));

    typedef int   IndexType;
    typedef float ValueType;
    typedef cusp::device_memory MemorySpace;

    cusp::csr_matrix<IndexType, ValueType, MemorySpace> A;
    size_t size = 1024;

    if (argc == 1)
    {
        // no input file was specified, generate an example
        cusp::gallery::poisson5pt(A, size, size);
    }
    else if (argc == 2)
    {
        // an input file was specified, read it from disk
        cusp::io::read_matrix_market_file(A, argv[1]);
    }

    std::cout << " Device ";
    RCM<cusp::device_memory>(A);

    std::cout << " Host ";
    RCM<cusp::host_memory>(A);

    std::cout << std::endl;

    return EXIT_SUCCESS;
}

