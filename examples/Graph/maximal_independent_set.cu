#include <cusp/csr_matrix.h>

#include <cusp/gallery/poisson.h>
#include <cusp/graph/maximal_independent_set.h>

int main(int argc, char*argv[])
{
    // create an empty sparse matrix structure (CSR format)
    cusp::csr_matrix<int,float,cusp::device_memory> G;

    // create a 2d Poisson problem on a 512x512 mesh
    cusp::gallery::poisson5pt(G, 512, 512);

    // create vector to contain MIS labels
    cusp::array1d<bool,cusp::device_memory> stencil(G.num_rows);

    // execute MIS computation on device
    cusp::graph::maximal_independent_set(G, stencil);

    return 0;
}

