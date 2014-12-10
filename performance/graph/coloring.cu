#include <cusp/csr_matrix.h>
#include <cusp/print.h>

#include <cusp/gallery/poisson.h>
#include <cusp/graph/connected_components.h>
#include <cusp/io/matrix_market.h>

#include "../timer.h"

template<typename MemorySpace, typename MatrixType>
void coloring(const MatrixType& G)
{
    typedef typename MatrixType::index_type IndexType;
    typedef cusp::csr_matrix<IndexType,IndexType,MemorySpace> GraphType;

    GraphType G_csr(G);

    timer t;

    size_t max_color = 0;
    size_t N = G.num_rows;
    cusp::array1d<IndexType,MemorySpace> colors(N, N-1);
    cusp::array1d<IndexType,MemorySpace> mark(N, std::numeric_limits<IndexType>::max());

    for(size_t vertex = 0; vertex < N; vertex++)
    {
        for(IndexType offset = G_csr.row_offsets[vertex]; offset < G_csr.row_offsets[vertex+1]; offset++)
        {
            IndexType neighbor = G_csr.column_indices[offset];
            mark[colors[neighbor]] = vertex;
        }

        size_t vertex_color = 0;
        while(vertex_color < max_color && mark[vertex_color] == vertex)
            vertex_color++;

        if(vertex_color == max_color)
            max_color++;

        colors[vertex] = vertex_color;
    }

    std::cout << "Coloring time     : " << t.milliseconds_elapsed() << " (ms)." << std::endl;

    cusp::array1d<IndexType,cusp::host_memory> color_counts(max_color);

    thrust::sort(colors.begin(), colors.end());

    thrust::reduce_by_key(colors.begin(),
                          colors.end(),
                          thrust::constant_iterator<int>(1),
                          thrust::make_discard_iterator(),
                          color_counts.begin());

    std::cout << "Number of colors : " << max_color << std::endl;

    cusp::print(color_counts);
}

int main(int argc, char*argv[])
{
    srand(time(NULL));

    typedef int   IndexType;
    typedef float ValueType;
    typedef cusp::host_memory MemorySpace;

    cusp::csr_matrix<IndexType, ValueType, MemorySpace> A;
    size_t size = 512;

    if (argc == 1)
    {
        // no input file was specified, generate an example
        std::cout << "Generated matrix (poisson5pt) ";
        cusp::gallery::poisson5pt(A, size, size);
    }
    else if (argc == 2)
    {
        // an input file was specified, read it from disk
        cusp::io::read_matrix_market_file(A, argv[1]);
        std::cout << "Read matrix (" << argv[1] << ") ";
    }

    std::cout << "with shape ("  << A.num_rows << "," << A.num_cols << ") and "
              << A.num_entries << " entries" << "\n\n";

    std::cout << " Host ";
    coloring<cusp::host_memory>(A);

    return EXIT_SUCCESS;
}

