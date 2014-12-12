#include <cusp/csr_matrix.h>
#include <cusp/gallery/poisson.h>
#include <cusp/graph/vertex_coloring.h>
#include <cusp/io/matrix_market.h>

#include "../../unittest/unittest.h"

template<typename MatrixType, typename ArrayType1, typename ArrayType2>
void gauss_seidel_indexed(const MatrixType& A,
                          ArrayType1&  x,
                          const ArrayType1&  b,
                          const ArrayType2& indices,
                          const int row_start,
                          const int row_stop,
                          const int row_step)
{
    typedef typename ArrayType1::value_type V;
    typedef typename ArrayType2::value_type I;

    for(int i = row_start; i != row_stop; i += row_step)
    {
        I inew  = indices[i];
        I start = A.row_offsets[inew];
        I end   = A.row_offsets[inew + 1];
        V rsum  = 0;
        V diag  = 0;

        for(I jj = start; jj < end; ++jj)
        {
            I j = A.column_indices[jj];
            if (inew == j)
            {
                diag = A.values[jj];
            }
            else
            {
                rsum += A.values[jj]*x[j];
            }
        }

        if (diag != 0)
        {
            x[inew] = (b[inew] - rsum)/diag;
        }
    }
}

int main(int argc, char*argv[])
{
    typedef cusp::host_memory MemorySpace;

    // create an empty sparse matrix structure (CSR format)
    cusp::csr_matrix<int, float, MemorySpace> A;

    size_t size = 4;

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

    cusp::counting_array<int> indices(A.num_rows);
    cusp::array1d<float,MemorySpace> x(A.num_rows, 0);
    cusp::array1d<float,MemorySpace> b(A.num_rows, 1);

    gauss_seidel_indexed(A, x, b, indices, 0, A.num_rows, 1);
    gauss_seidel_indexed(A, x, b, indices, A.num_rows-1, -1, -1);

    cusp::array1d<int,MemorySpace> colors(A.num_rows);
    int max_colors = cusp::graph::vertex_coloring(A, colors);

    cusp::array1d<int,MemorySpace> sorted_colors(colors);
    cusp::array1d<int,MemorySpace> color_counts(max_colors + 1);
    cusp::array1d<int,MemorySpace> permutation1(A.num_rows);
    thrust::sequence(permutation1.begin(), permutation1.end());

    thrust::sort_by_key(sorted_colors.begin(), sorted_colors.end(), permutation1.begin());
    thrust::reduce_by_key(sorted_colors.begin(),
                          sorted_colors.end(),
                          thrust::constant_iterator<int>(1),
                          thrust::make_discard_iterator(),
                          color_counts.begin());
    thrust::exclusive_scan(color_counts.begin(), color_counts.end(), color_counts.begin(), 0);

    cusp::print(color_counts);

    cusp::array1d<int,MemorySpace> permutation2(A.num_rows);
    thrust::scatter(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(A.num_rows),
                    permutation1.begin(), permutation2.begin());

    cusp::array1d<float,MemorySpace> x_color(A.num_rows, 0);
    for(int i = 0; i < max_colors; i++)
        gauss_seidel_indexed(A, x_color, b, permutation2, color_counts[i], color_counts[i+1], 1);
    for(int i = max_colors; i > 0; i--)
        gauss_seidel_indexed(A, x_color, b, permutation2, color_counts[i-1], color_counts[i], 1);

    /* ASSERT_ALMOST_EQUAL(x, x_color); */

    if(A.num_rows <= 16)
    {
      cusp::print(x);
      cusp::print(x_color);
    }

    return 0;
}
