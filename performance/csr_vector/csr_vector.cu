#include <cusp/csr_matrix.h>
#include <cusp/io/matrix_market.h>

#include <thrust/sequence.h>
#include <thrust/fill.h>

#include <iostream>

#include <cusp/system/cuda/detail/multiply/csr_vector_spmv.h>
#include "../timer.h"

template <unsigned int THREADS_PER_VECTOR, typename MatrixType, typename VectorType1, typename VectorType2>
void perform_spmv(const MatrixType& csr,
                  const VectorType1& x,
                        VectorType2& y)
{
    typedef typename MatrixType::index_type IndexType;
    typedef typename MatrixType::value_type ValueType;

    typedef typename MatrixType::row_offsets_array_type::const_iterator     RowIterator;
    typedef typename MatrixType::column_indices_array_type::const_iterator  ColumnIterator;
    typedef typename MatrixType::values_array_type::const_iterator          ValueIterator1;

    typedef typename VectorType1::const_iterator                            ValueIterator2;
    typedef typename VectorType2::iterator                                  ValueIterator3;

    const unsigned int VECTORS_PER_BLOCK  = 128 / THREADS_PER_VECTOR;
    const unsigned int THREADS_PER_BLOCK  = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

    const unsigned int MAX_BLOCKS = 16 * 1024;
    const unsigned int NUM_BLOCKS = std::min(MAX_BLOCKS, static_cast<unsigned int>((csr.num_rows + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK));

    cusp::system::cuda::detail::spmv_csr_vector_kernel<RowIterator, ColumnIterator, ValueIterator1, ValueIterator2, ValueIterator3,
                                                       UnaryFunction, BinaryFunction1, BinaryFunction2,
                                                       VECTORS_PER_BLOCK, THREADS_PER_VECTOR>
                                                      <<<NUM_BLOCKS, THREADS_PER_BLOCK, 0, s>>>
        (csr.num_rows, csr.row_offsets.begin(), csr.column_indices.begin(), csr.values.begin());
}

template <unsigned int ThreadsPerVector, typename IndexType, typename ValueType>
float benchmark_matrix(const cusp::csr_matrix<IndexType,ValueType,cusp::device_memory>& csr)
{
    const size_t num_iterations = 100;

    cusp::array1d<ValueType, cusp::device_memory> x(csr.num_cols);
    cusp::array1d<ValueType, cusp::device_memory> y(csr.num_rows);

    // warmup
    perform_spmv<ThreadsPerVector>(csr, x, y);

    // time several SpMV iterations
    timer t;
    for(size_t i = 0; i < num_iterations; i++)
        perform_spmv<ThreadsPerVector>(csr, x, y);
    cudaThreadSynchronize();

    float sec_per_iteration = t.seconds_elapsed() / num_iterations;
    float gflops = 2.0 * (csr.num_entries/sec_per_iteration) / 1e9;

    return gflops;
}


template <typename IndexType, typename ValueType>
void make_synthetic_example(const IndexType N, const IndexType D,
                            cusp::csr_matrix<IndexType, ValueType, cusp::device_memory>& csr)
{
    // create ELL matrix with D diagonals
    cusp::ell_matrix<IndexType, ValueType, cusp::host_memory> ell(N, D, N * D, D);
    for(IndexType i = 0; i < N; i++)
    {
        for(IndexType j = 0; j < D; j++)
        {
            ell.column_indices(i,j) = j;
            ell.values(i,j) = 1.0;
        }
    }

    // convert to CSR
    csr = ell;
}


int main(int argc, char** argv)
{
    typedef int   IndexType;
    typedef float ValueType;


    if (argc == 1)
    {
        // matrix varies along rows, # of threads per vector varies along column
        printf("matrix      , nnz per row,       2,       4,       8,      16,      32,\n");

        const IndexType N = 320 * 1000;
        const IndexType max_diagonals = 64;

        for(IndexType D = 1; D <= max_diagonals; D++)
        {
            cusp::csr_matrix<IndexType, ValueType, cusp::device_memory> csr;
            make_synthetic_example(N, D, csr);
            printf("dense_%02d    ,    %8.2f,", (int) D, (float) csr.num_entries / (float) csr.num_rows);
            printf("  %5.4f,", benchmark_matrix< 2, IndexType, ValueType>(csr));
            printf("  %5.4f,", benchmark_matrix< 4, IndexType, ValueType>(csr));
            printf("  %5.4f,", benchmark_matrix< 8, IndexType, ValueType>(csr));
            printf("  %5.4f,", benchmark_matrix<16, IndexType, ValueType>(csr));
            printf("  %5.4f,", benchmark_matrix<32, IndexType, ValueType>(csr));
            printf("\n");
        }
    }
    else
    {
        // matrix varies along rows, # of threads per vector varies along column
        printf("matrix              , nnz per row,       2,       4,       8,      16,      32,\n");

        for(int i = 1; i < argc; i++)
        {
            cusp::csr_matrix<IndexType, ValueType, cusp::device_memory> csr;
            cusp::io::read_matrix_market_file(csr, std::string(argv[i]));
            printf("%20s,    %8.2f,", argv[i], (float) csr.num_entries / (float) csr.num_rows);
            printf("  %5.4f,", benchmark_matrix< 2, IndexType, ValueType>(csr));
            printf("  %5.4f,", benchmark_matrix< 4, IndexType, ValueType>(csr));
            printf("  %5.4f,", benchmark_matrix< 8, IndexType, ValueType>(csr));
            printf("  %5.4f,", benchmark_matrix<16, IndexType, ValueType>(csr));
            printf("  %5.4f,", benchmark_matrix<32, IndexType, ValueType>(csr));
            printf("\n");
        }
    }

    return 0;
}
