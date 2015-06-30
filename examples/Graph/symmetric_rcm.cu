#include <cusp/csr_matrix.h>
#include <cusp/permutation_matrix.h>

#include <cusp/gallery/poisson.h>
#include <cusp/graph/symmetric_rcm.h>

int main(int argc, char*argv[])
{
    // create an empty sparse matrix structure (CSR format)
    cusp::csr_matrix<int,float,cusp::device_memory> G;

    // create a 2d Poisson problem on a 1024x1024 mesh
    cusp::gallery::poisson5pt(G, 1024, 1024);

    // create an empty permutation matrix
    cusp::permutation_matrix<int,cusp::device_memory> P(G.num_rows);

    // compute RCM permutation on device
    cusp::graph::symmetric_rcm(G, P);

    // symmetrically permute G
    P.symmetric_permute(G);

    return 0;
}

