#include <cusp/dia_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/convert.h>
#include <cusp/io.h>

#include <thrust/sequence.h>
#include <thrust/fill.h>

#include <iostream>

#include <cusp/detail/device/spmv.h>
#include "timer.h"


template <bool UseCache, unsigned int THREADS_PER_VECTOR, typename IndexType, typename ValueType>
void perform_spmv(const cusp::csr_matrix<IndexType,ValueType,cusp::device>& csr, 
                  const ValueType * x, 
                        ValueType * y)
{
    const unsigned int VECTORS_PER_BLOCK  = 128 / THREADS_PER_VECTOR;
    const unsigned int THREADS_PER_BLOCK  = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

    //const unsigned int MAX_BLOCKS = MAX_THREADS / THREADS_PER_BLOCK;
    const unsigned int MAX_BLOCKS = thrust::experimental::arch::max_active_blocks(cusp::detail::device::spmv_csr_vector_kernel<IndexType, ValueType, VECTORS_PER_BLOCK, THREADS_PER_VECTOR, UseCache>, THREADS_PER_BLOCK, (size_t) 0);
    const unsigned int NUM_BLOCKS = std::min(MAX_BLOCKS, DIVIDE_INTO(csr.num_rows, VECTORS_PER_BLOCK));
    
    if (UseCache)
        bind_x(x);

    cusp::detail::device::spmv_csr_vector_kernel<IndexType, ValueType, VECTORS_PER_BLOCK, THREADS_PER_VECTOR, UseCache> <<<NUM_BLOCKS, THREADS_PER_BLOCK>>> 
        (csr.num_rows,
         thrust::raw_pointer_cast(&csr.row_offsets[0]),
         thrust::raw_pointer_cast(&csr.column_indices[0]),
         thrust::raw_pointer_cast(&csr.values[0]),
         x, y);

    if (UseCache)
        unbind_x(x);
}
 
template <typename IndexType, typename ValueType, unsigned int ThreadsPerVector>
void benchmark_csr_vector(size_t num_iterations = 100)
{
    std::cout << ThreadsPerVector << "\t| ";

    const IndexType N = 16 * 1000; // for alignment

    cusp::vector<ValueType, cusp::device> x(N);
    cusp::vector<ValueType, cusp::device> y(N);

    for(IndexType D = 1; D < 64; D += 2)
    {
        // create banded matrix with D diagonals
        const IndexType NNZ = N + (D-1) * N - D*D;
        cusp::dia_matrix<IndexType, ValueType, cusp::host> dia(N, N, NNZ, D, N);
        thrust::sequence(dia.diagonal_offsets.begin(), dia.diagonal_offsets.end(), -D);
        thrust::fill(dia.values.begin(), dia.values.end(), 1);

        // convert to CSR
        cusp::csr_matrix<IndexType, ValueType, cusp::host> h_csr;
        cusp::convert(h_csr, dia);
        cusp::csr_matrix<IndexType, ValueType, cusp::device> d_csr(h_csr);
    
        // time several SpMV iterations
        timer t;
        for(size_t i = 0; i < num_iterations; i++)
            perform_spmv<true, ThreadsPerVector>(d_csr, thrust::raw_pointer_cast(&x[0]), thrust::raw_pointer_cast(&y[0]));
        cudaThreadSynchronize();

        float sec_per_iteration = t.seconds_elapsed() / num_iterations;
        float gflops = (NNZ/sec_per_iteration) / 1e9;

        printf("%4.1f ", gflops);
    }

    std::cout << std::endl;
}


int main(int argc, char** argv)
{
    benchmark_csr_vector<int, float,  2>();
    benchmark_csr_vector<int, float,  4>();
    benchmark_csr_vector<int, float,  8>();
    benchmark_csr_vector<int, float, 16>();
    benchmark_csr_vector<int, float, 32>();

    return 0;
}

