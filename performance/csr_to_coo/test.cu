#include <cusp/gallery/poisson.h>
#include <cusp/csr_matrix.h>
#include <cusp/io/matrix_market.h>
#include <cusp/blas.h>

#include <thrust/binary_search.h>

#include <iostream>
#include <string>
#include <map>
#include <cmath>
#include <stdio.h>

#include "../timer.h"

template <typename IndexType, typename ValueType>
void csr_to_coo(const cusp::csr_matrix<IndexType, ValueType, cusp::device_memory>& A,
                      cusp::coo_matrix<IndexType, ValueType, cusp::device_memory>& B)
{
    B.resize(A.num_rows, A.num_cols, A.num_entries);
            
    // compute the row index for each matrix entry
    thrust::upper_bound(A.row_offsets.begin() + 1,
                        A.row_offsets.end(),
                        thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(A.num_entries),
                        B.row_indices.begin());

    B.column_indices = A.column_indices;
    B.values         = A.values;
}


template <typename IndexType, typename ValueType>
void benchmark_host(cusp::csr_matrix<IndexType, ValueType, cusp::host_memory>& H)
{
    unsigned int N = 100;
    
    timer t;

    for(unsigned int i = 0; i < N; i++)
    {
        cusp::coo_matrix<IndexType,ValueType,cusp::host_memory> A(H);
    }

    float time = t.milliseconds_elapsed() / N;

    printf("host conversion:    %8.4f msecs\n", time);
}


template <typename IndexType, typename ValueType>
void benchmark_device(cusp::csr_matrix<IndexType, ValueType, cusp::host_memory>& H)
{
    cusp::csr_matrix<IndexType, ValueType, cusp::device_memory> D(H);

    unsigned int N = 100;
    
    timer t;

    for(unsigned int i = 0; i < N; i++)
    {
        cusp::coo_matrix<IndexType,ValueType,cusp::device_memory> A;
        csr_to_coo(D,A);
    }

    float time = t.milliseconds_elapsed() / N;

    printf("device conversion:  %8.4f msecs\n", time);
}

int main(void)
{
    cudaSetDevice(0);

    typedef int    IndexType;
    typedef float  ValueType;

    cusp::csr_matrix<IndexType, ValueType, cusp::host_memory> A;
    
    //cusp::io::read_matrix_market_file(A, "A.mtx");
    //std::cout << "loaded matrix with shape (" << A.num_rows << "," << A.num_cols << ") and " << A.num_entries << " entries" << "\n\n";

    cusp::gallery::poisson5pt(A, 1000, 100);
    
    std::cout << "matrix has shape (" << A.num_rows << "," << A.num_cols << ") and " << A.num_entries << " entries" << "\n\n";

    benchmark_host(A);
    benchmark_device(A);

    return 0;
}

