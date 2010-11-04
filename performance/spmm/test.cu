#include <cusp/gallery/poisson.h>
#include <cusp/io/matrix_market.h>
#include <cusp/multiply.h>

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/dia_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

#include <iostream>
#include <stdio.h>

#include "../timer.h"


template <typename MatrixType, typename InputType>
float time_spmm(const InputType& A,
                const InputType& B)
{
    unsigned int N = 10;

    MatrixType A_;
    MatrixType B_;

    try
    {
        A_ = A;
        B_ = B;
    }
    catch (cusp::format_conversion_exception)
    {
        return -1;
    }
    
    timer t;

    for(unsigned int i = 0; i < N; i++)
    {
        MatrixType C_;
        cusp::multiply(A_, B_, C_);
    }

    return t.milliseconds_elapsed() / N;
}

template <typename MemorySpace, typename InputType>
void for_each_type(const InputType& A,
                   const InputType& B)
{
    typedef typename InputType::index_type I;
    typedef typename InputType::value_type V;

    typedef cusp::coo_matrix<I,V,MemorySpace> COO;
    typedef cusp::csr_matrix<I,V,MemorySpace> CSR;
    typedef cusp::dia_matrix<I,V,MemorySpace> DIA;
    typedef cusp::ell_matrix<I,V,MemorySpace> ELL;
    typedef cusp::hyb_matrix<I,V,MemorySpace> HYB;

    printf("  Format   |\n");
    printf("    COO    | %9.2f\n", time_spmm<COO>(A,B));
    printf("    CSR    | %9.2f\n", time_spmm<CSR>(A,B));
    printf("    DIA    | %9.2f\n", time_spmm<DIA>(A,B));
    printf("    ELL    | %9.2f\n", time_spmm<ELL>(A,B));
    printf("    HYB    | %9.2f\n", time_spmm<HYB>(A,B));
}

int main(int argc, char ** argv)
{
    cudaSetDevice(0);

    typedef int    IndexType;
    typedef float  ValueType;

    cusp::csr_matrix<IndexType, ValueType, cusp::host_memory> A;
    cusp::csr_matrix<IndexType, ValueType, cusp::host_memory> B;

    if (argc == 1)
    {
        // no input file was specified, generate an example
        cusp::gallery::poisson5pt(A, 200, 200);
        cusp::gallery::poisson5pt(B, 200, 200);
    }
    else if (argc == 3)
    {
        // input files were specified, read them from disk
        cusp::io::read_matrix_market_file(A, argv[1]);
        cusp::io::read_matrix_market_file(B, argv[2]);
    }
    
    std::cout << "Input matrix A has shape (" << A.num_rows << "," << A.num_cols << ") and " << A.num_entries << " entries" << "\n";
    std::cout << "             B has shape (" << B.num_rows << "," << B.num_cols << ") and " << B.num_entries << " entries" << "\n\n";
   
    printf("Host Sparse Matrix-Matrix Multiply (milliseconds per multiplication)\n");
    for_each_type<cusp::host_memory>(A,B);
   
    printf("\n\n");

    printf("Device Sparse Matrix-Matrix Multiply (milliseconds per multiplication)\n");
    for_each_type<cusp::device_memory>(A,B);

    return 0;
}

