#include <cusp/csr_matrix.h>
#include <cusp/print.h>

#include <cusp/gallery/poisson.h>
#include <cusp/graph/detail/rcm.inl>
#include <cusp/io/matrix_market.h>

#include <thrust/functional.h>
#include "../timer.h"

template<typename T>
struct absolute_value : public thrust::unary_function<T,T>
{
  __host__ __device__ T operator()(const T &x) const
  {
    return x < T(0) ? -x : x;
  }
};


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

    {
        cusp::array1d<IndexType, MemorySpace> row_indices(G.row_indices);
        cusp::array1d<IndexType, MemorySpace> min_column(G.num_rows, 0);
        cusp::array1d<IndexType, MemorySpace> max_column(G.num_rows, 0);
        cusp::array1d<IndexType, MemorySpace> rowwise_bandwidth(G.num_rows, 0);

        thrust::reduce_by_key(row_indices.begin(),
                              row_indices.end(),
                              G_bfs.column_indices.begin(),
                              thrust::make_discard_iterator(),
                              min_column.begin(),
                              thrust::equal_to<IndexType>(),
                              thrust::minimum<IndexType>());
        thrust::reduce_by_key(row_indices.begin(),
                              row_indices.end(),
                              G_bfs.column_indices.begin(),
                              thrust::make_discard_iterator(),
                              max_column.begin(),
                              thrust::equal_to<IndexType>(),
                              thrust::maximum<IndexType>());

        thrust::transform(max_column.begin(), max_column.end(), min_column.begin(), rowwise_bandwidth.begin(), thrust::minus<IndexType>());
        IndexType bandwidth = *thrust::max_element(rowwise_bandwidth.begin(), rowwise_bandwidth.end()) + 1;
        /*cusp::array1d<IndexType, MemorySpace> row_indices(G.num_entries);
        cusp::array1d<IndexType, MemorySpace> diff(G.num_entries);

	cusp::detail::offsets_to_indices(G_bfs.row_offsets, row_indices);
        thrust::transform(row_indices.begin(), row_indices.end(), G_bfs.column_indices.begin(), diff.begin(), thrust::minus<IndexType>());
        IndexType bandwidth = thrust::transform_reduce(diff.begin(), diff.end(), absolute_value<IndexType>(), IndexType(-1), thrust::maximum<IndexType>());*/
        std::cout << "Bandwidth after RCM : " << bandwidth << std::endl;
    }
}

int main(int argc, char*argv[])
{
    srand(time(NULL));

    typedef int   IndexType;
    typedef float ValueType;
    typedef cusp::device_memory MemorySpace;

    cusp::coo_matrix<IndexType, ValueType, MemorySpace> A;
    size_t size = 4;

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

    {
        cusp::array1d<IndexType, MemorySpace> min_column(A.num_rows, 0);
        cusp::array1d<IndexType, MemorySpace> max_column(A.num_rows, 0);
        cusp::array1d<IndexType, MemorySpace> rowwise_bandwidth(A.num_rows, 0);

        thrust::reduce_by_key(A.row_indices.begin(),
                              A.row_indices.end(),
                              A.column_indices.begin(),
                              thrust::make_discard_iterator(),
                              min_column.begin(),
                              thrust::equal_to<IndexType>(),
                              thrust::minimum<IndexType>());
        thrust::reduce_by_key(A.row_indices.begin(),
                              A.row_indices.end(),
                              A.column_indices.begin(),
                              thrust::make_discard_iterator(),
                              max_column.begin(),
                              thrust::equal_to<IndexType>(),
                              thrust::maximum<IndexType>());

        thrust::transform(max_column.begin(), max_column.end(), min_column.begin(), rowwise_bandwidth.begin(), thrust::minus<IndexType>());
        IndexType bandwidth = *thrust::max_element(rowwise_bandwidth.begin(), rowwise_bandwidth.end()) + 1;
        std::cout << "Bandwidth before RCM : " << bandwidth << std::endl;
    }

    std::cout << " Device ";
    RCM<cusp::device_memory>(A);

    std::cout << " Host ";
    RCM<cusp::host_memory>(A);

    std::cout << std::endl;

    return EXIT_SUCCESS;
}

