#include <cusp/csr_matrix.h>
#include <cusp/io.h>

#include <iostream>
#include <string>
#include <map>
#include <cmath>
#include <limits>

#include <cusp/host/spmv.h>
#include <cusp/device/spmv.h>

#include "timer.h"

std::map<std::string, std::string> args;


template <typename T>
T l2_error(size_t N, const T * a, const T * b)
{
    T numerator   = 0;
    T denominator = 0;
    for(size_t i = 0; i < N; i++)
    {
        numerator   += (a[i] - b[i]) * (a[i] - b[i]);
        denominator += (b[i] * b[i]);
    }

    return numerator/denominator;
}


// HostTestMatrix is a WAR because we can't convert directly
template <typename TestMatrix, typename HostTestMatrix, typename HostMatrix>
void test(std::string name, HostMatrix& host_matrix, double seconds = 3.0, size_t min_iterations = 100, size_t max_iterations = 500)
{
    typedef typename TestMatrix::index_type   IndexType;
    typedef typename TestMatrix::value_type   ValueType;
    typedef typename TestMatrix::memory_space MemorySpace;

    const IndexType M = host_matrix.num_rows;
    const IndexType N = host_matrix.num_cols;

    // convert host_matrix to TestMatrix format
    HostTestMatrix host_test_matrix;
   
    try 
    {
        cusp::convert_matrix(host_test_matrix, host_matrix);
    } 
    catch (cusp::format_conversion_exception)
    {
        // conversion failed
        return;
    }
    TestMatrix test_matrix;
    cusp::convert_matrix(test_matrix, host_test_matrix);
    cusp::deallocate_matrix(host_test_matrix);

    // create host input (x) and output (y) vectors
    ValueType * host_x = cusp::new_array<ValueType, cusp::host_memory>(N);
    ValueType * host_y = cusp::new_array<ValueType, cusp::host_memory>(M);
    for(IndexType i = 0; i < N; i++) host_x[i] = 1.0; //.5; //1.0/3.0; //rand() % 100; //(int(i % 21) - 10)
    for(IndexType i = 0; i < M; i++) host_y[i] = 0;

    // create test input (x) and output (y) vectors
    ValueType * test_x = cusp::new_array<ValueType, MemorySpace>(N);
    ValueType * test_y = cusp::new_array<ValueType, MemorySpace>(M);
    cusp::memcpy_array<ValueType, MemorySpace, cusp::host_memory>(test_x, host_x, N);
    cusp::memcpy_array<ValueType, MemorySpace, cusp::host_memory>(test_y, host_y, M);

    // compute SpMV on host and device
    cusp::host::spmv(host_matrix, host_x, host_y);
    cusp::device::spmv(test_matrix, test_x, test_y);  // TODO generalize this

    // compare results
    ValueType * test_y_copy = cusp::new_array<ValueType, cusp::host_memory>(M);
    cusp::memcpy_array<ValueType, cusp::host_memory, MemorySpace>(test_y_copy, test_y, M);
    double error = l2_error(M, test_y_copy, host_y);
    cusp::delete_array<ValueType, cusp::host_memory>(test_y_copy);
   
    // warmup
    timer time_one_iteration;
    cusp::device::spmv(test_matrix, test_x, test_y);  // TODO generalize this
    cudaThreadSynchronize();
    double estimated_time = time_one_iteration.seconds_elapsed();
    
    // determine # of seconds dynamically
    size_t num_iterations;
    if (estimated_time == 0)
        num_iterations = max_iterations;
    else
        num_iterations = std::min(max_iterations, std::max(min_iterations, (size_t) (seconds / estimated_time)) ); 
    
    // time several SpMV iterations
    timer t;
    for(size_t i = 0; i < num_iterations; i++)
        cusp::device::spmv(test_matrix, test_x, test_y);  // TODO generalize this
    cudaThreadSynchronize();
    double msec_per_iteration = t.milliseconds_elapsed() / (double) num_iterations;
    double sec_per_iteration = msec_per_iteration / 1000.0;
    double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double) host_matrix.num_entries / sec_per_iteration) / 1e9;
    double GBYTEs = 0.0;
    //double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double) bytes_per_spmv(sp_host) / sec_per_iteration) / 1e9;
    
    printf("\t%s: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s) [L2 error %.10f]\n", name.c_str(), msec_per_iteration, GFLOPs, GBYTEs, error); 
    
    // clean up our work space
    cusp::delete_array<ValueType, cusp::host_memory>(host_x);
    cusp::delete_array<ValueType, cusp::host_memory>(host_y);
    cusp::delete_array<ValueType, MemorySpace>(test_x);
    cusp::delete_array<ValueType, MemorySpace>(test_y);

    cusp::deallocate_matrix(test_matrix);
}


std::string process_args(int argc, char ** argv)
{
    std::string filename;

    for(int i = 1; i < argc; i++)
    {
        std::string arg(argv[i]);

        if (arg.substr(0,2) == "--")
        {   
            std::string::size_type n = arg.find('=',2);
            args[arg.substr(2,n)] = arg.substr(n);
        }
        else
        {
            filename = arg;
        }
    }

    return filename;
}

int main(int argc, char** argv)
{
    std::string filename = process_args(argc, argv);

    // load a matrix stored in MatrixMarket format
    cusp::csr_matrix<int, float, cusp::host_memory> host_matrix;
    cusp::load_matrix_market_file(host_matrix, filename);
   
    std::cout << "Read matrix (" << filename << ") with shape ("
              << host_matrix.num_rows << "," << host_matrix.num_cols << ") and "
              << host_matrix.num_entries << " entries" << std::endl;

    test< cusp::coo_matrix<int, float, cusp::device_memory>, cusp::coo_matrix<int, float, cusp::host_memory> >("coo", host_matrix);
    test< cusp::csr_matrix<int, float, cusp::device_memory>, cusp::csr_matrix<int, float, cusp::host_memory> >("csr", host_matrix);
    test< cusp::dia_matrix<int, float, cusp::device_memory>, cusp::dia_matrix<int, float, cusp::host_memory> >("dia", host_matrix);
    test< cusp::ell_matrix<int, float, cusp::device_memory>, cusp::ell_matrix<int, float, cusp::host_memory> >("ell", host_matrix);
    test< cusp::hyb_matrix<int, float, cusp::device_memory>, cusp::hyb_matrix<int, float, cusp::host_memory> >("hyb", host_matrix);

    cusp::deallocate_matrix(host_matrix);

    return 0;
}

