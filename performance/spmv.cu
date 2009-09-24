#include <cusp/csr_matrix.h>
#include <cusp/io.h>

#include <iostream>
#include <string>
#include <map>
#include <cmath>
#include <limits>

#include <cusp/host/spmv.h>
#include <cusp/device/spmv.h>

#include "bytes_per_spmv.h"
#include "timer.h"
#include "gallery.h"

typedef std::map<std::string, std::string> ArgumentMap;
ArgumentMap args;


void usage(int argc, char** argv)
{
    std::cout << "Usage:\n";
    std::cout << "\t" << argv[0] << "\n";
    std::cout << "\t" << argv[0] << " my_matrix.mtx\n";
    std::cout << "\t" << argv[0] << " my_matrix.mtx --device=1\n";
    std::cout << "\t" << argv[0] << " my_matrix.mtx --value_type=double\n";
    std::cout << "Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n"; 
    std::cout << "      If no matrix file is provided then a simple example is created.\n";  
}


void set_device(int device_id)
{
    cudaSetDevice(device_id);
}


void list_devices(void)
{
    int deviceCount;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0)
        std::cout << "There is no device supporting CUDA" << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev));

        if (dev == 0) {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                std::cout << "There is no device supporting CUDA." << std::endl;
            else if (deviceCount == 1)
                std::cout << "There is 1 device supporting CUDA" << std:: endl;
            else
                std::cout << "There are " << deviceCount <<  " devices supporting CUDA" << std:: endl;
        }

        std::cout << "\nDevice " << dev << ": \"" << deviceProp.name << "\"" << std::endl;
        std::cout << "  Major revision number:                         " << deviceProp.major << std::endl;
        std::cout << "  Minor revision number:                         " << deviceProp.minor << std::endl;
        std::cout << "  Total amount of global memory:                 " << deviceProp.totalGlobalMem << " bytes" << std::endl;
    }
    std::cout << std::endl;
}


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

    // create host input (x) and output (y) vectors
    ValueType * host_x = cusp::new_array<ValueType, cusp::host_memory>(N);
    ValueType * host_y = cusp::new_array<ValueType, cusp::host_memory>(M);
    for(IndexType i = 0; i < N; i++) host_x[i] = (int(i % 21) - 10);
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
    cusp::device::spmv_tex(test_matrix, test_x, test_y);  // TODO generalize this
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
        cusp::device::spmv_tex(test_matrix, test_x, test_y);  // TODO generalize this
    cudaThreadSynchronize();
    double msec_per_iteration = t.milliseconds_elapsed() / (double) num_iterations;
    double sec_per_iteration = msec_per_iteration / 1000.0;
    double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double) host_matrix.num_entries / sec_per_iteration) / 1e9;
    double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double) bytes_per_spmv(host_test_matrix) / sec_per_iteration) / 1e9;
   
    printf("\t%s: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s) [L2 error %.8f]\n", name.c_str(), msec_per_iteration, GFLOPs, GBYTEs, error); 
    
    // clean up our work space
    cusp::deallocate_matrix(host_test_matrix);
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

            if (n == std::string::npos)
                args[arg.substr(2)] = std::string();              // (key)
            else
                args[arg.substr(2, n - 2)] = arg.substr(n + 1);   // (key,value)
        }
        else
        {
            filename = arg;
        }
    }

    return filename;
}


template <typename IndexType, typename ValueType>
void test_all_formats(std::string& filename)
{
    int device_id  = args.count("device") ? atoi(args["device"].c_str()) :  0;
    set_device(device_id);
    list_devices();

    std::cout << "Running on Device " << device_id << "\n\n";
    
    // load a matrix stored in MatrixMarket format
    cusp::csr_matrix<IndexType, ValueType, cusp::host_memory> host_matrix;

    if (filename == "")
    {
        std::cout << "Generated matrix (laplace_2d) ";
        laplacian_5pt(host_matrix, 1024);
    }
    else
    {
        cusp::load_matrix_market_file(host_matrix, filename);
        std::cout << "Read matrix (" << filename << ") ";
    }
        
    std::cout << "with shape ("  << host_matrix.num_rows << "," << host_matrix.num_cols << ") and "
              << host_matrix.num_entries << " entries" << "\n\n";

    test< cusp::coo_matrix<IndexType, ValueType, cusp::device_memory>, cusp::coo_matrix<IndexType, ValueType, cusp::host_memory> >("coo", host_matrix);
    test< cusp::csr_matrix<IndexType, ValueType, cusp::device_memory>, cusp::csr_matrix<IndexType, ValueType, cusp::host_memory> >("csr", host_matrix);
    test< cusp::dia_matrix<IndexType, ValueType, cusp::device_memory>, cusp::dia_matrix<IndexType, ValueType, cusp::host_memory> >("dia", host_matrix);
    test< cusp::ell_matrix<IndexType, ValueType, cusp::device_memory>, cusp::ell_matrix<IndexType, ValueType, cusp::host_memory> >("ell", host_matrix);
    test< cusp::hyb_matrix<IndexType, ValueType, cusp::device_memory>, cusp::hyb_matrix<IndexType, ValueType, cusp::host_memory> >("hyb", host_matrix);

    cusp::deallocate_matrix(host_matrix);
}

int main(int argc, char** argv)
{
    std::string filename = process_args(argc, argv);

    if (args.count("help")) usage(argc, argv);

    // select ValueType
    std::string value_type = args.count("value_type") ? args["value_type"] : "float";
    std::cout << "\nComputing SpMV with \'" << value_type << "\' values.\n\n";

    if (value_type == "float")
    {
        test_all_formats<int,float>(filename);
    }
    else if (value_type == "double")
    {
#if defined(CUDA_NO_SM_13_DOUBLE_INTRINSICS)
        std::cerr << "ERROR: Support for \'double\' requires SM 1.3 or greater (recompile with --arch=sm_13)\n\n";
#else
        test_all_formats<int,double>(filename);
#endif
    }
    else
    {
        std::cerr << "ERROR: Unsupported type \'" << value_type << "\'\n\n";
    }

    return 0;
}

