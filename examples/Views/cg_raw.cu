#include <cusp/csr_matrix.h>
#include <cusp/monitor.h>
#include <cusp/multiply.h>
#include <cusp/print.h>
#include <cusp/krylov/cg.h>

#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#else
#include <cstdlib>
#include <cstring>
#endif

// This example shows how to solve a linear system A * x = b
// where the matrix A and vectors x and b are stored in
// "raw" memory on the device.  The matrix A will be wrapped
// with a coo_matrix_view while the vectors x and b will be
// wrapped with array1d_views.
//
// Views allow you to interface Cusp's solvers with
// data that is managed externally without needing to
// copy the data into a Cusp matrix container.
//
//  Example Matrix:
//   [ 2 -1  0  0]
//   [-1  2 -1  0]
//   [ 0 -1  2 -1]
//   [ 0  0 -1  2]


int main(void)
{
    // COO format in host memory
    int   host_I[10] = { 0, 0, 1, 1, 1, 2, 2, 2, 3, 3}; // COO row indices
    int   host_J[10] = { 0, 1, 0, 1, 2, 1, 2, 3, 2, 3}; // COO column indices
    float host_V[10] = { 2,-1,-1, 2,-1,-1, 2,-1,-1, 2}; // COO values

    // x and b arrays in host memory
    float host_x[4] = {0,0,0,0};
    float host_b[4] = {1,2,2,1};

    // allocate memory
    int   * device_I;
    int   * device_J;
    float * device_V;
    float * device_x;
    float * device_b;
#ifdef __CUDACC__
    cudaMalloc(&device_I, 10 * sizeof(int));
    cudaMalloc(&device_J, 10 * sizeof(int));
    cudaMalloc(&device_V, 10 * sizeof(float));
    cudaMalloc(&device_x,  4 * sizeof(float));
    cudaMalloc(&device_b,  4 * sizeof(float));

    cudaMemcpy(device_I, host_I, 10 * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(device_J, host_J, 10 * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(device_V, host_V, 10 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_x, host_x,  4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b,  4 * sizeof(float), cudaMemcpyHostToDevice);

    thrust::device_ptr<int>   wrapped_I(device_I);
    thrust::device_ptr<int>   wrapped_J(device_J);
    thrust::device_ptr<float> wrapped_V(device_V);
    thrust::device_ptr<float> wrapped_x(device_x);
    thrust::device_ptr<float> wrapped_b(device_b);

    typedef typename cusp::array1d_view< thrust::device_ptr<int>   > IndexArrayView;
    typedef typename cusp::array1d_view< thrust::device_ptr<float> > ValueArrayView;
#else
    device_I = (int*)  malloc(10 * sizeof(int));
    device_J = (int*)  malloc(10 * sizeof(int));
    device_V = (float*)malloc(10 * sizeof(float));
    device_x = (float*)malloc( 4 * sizeof(float));
    device_b = (float*)malloc( 4 * sizeof(float));

    memcpy(device_I, host_I, 10 * sizeof(int));
    memcpy(device_J, host_J, 10 * sizeof(int));
    memcpy(device_V, host_V, 10 * sizeof(float));
    memcpy(device_x, host_x,  4 * sizeof(float));
    memcpy(device_b, host_b,  4 * sizeof(float));

    int*   wrapped_I = device_I;
    int*   wrapped_J = device_J;
    float* wrapped_V = device_V;
    float* wrapped_x = device_x;
    float* wrapped_b = device_b;

    typedef typename cusp::array1d_view<int*>   IndexArrayView;
    typedef typename cusp::array1d_view<float*> ValueArrayView;
#endif

    IndexArrayView row_indices   (wrapped_I, wrapped_I + 10);
    IndexArrayView column_indices(wrapped_J, wrapped_J + 10);
    ValueArrayView values        (wrapped_V, wrapped_V + 10);
    ValueArrayView x             (wrapped_x, wrapped_x + 4);
    ValueArrayView b             (wrapped_b, wrapped_b + 4);

    typedef cusp::coo_matrix_view<IndexArrayView, IndexArrayView, ValueArrayView> View;

    View A(4, 4, 10, row_indices, column_indices, values);

    cusp::monitor<float> monitor(b, 100, 1e-5, 0, true);

    cusp::krylov::cg(A, x, b, monitor);

    // copy the solution back to the host
#ifdef __CUDACC__
    cudaMemcpy(host_x, device_x, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_I);
    cudaFree(device_J);
    cudaFree(device_V);
    cudaFree(device_x);
    cudaFree(device_b);
#else
    memcpy(host_x, device_x, 4 * sizeof(float));
    free(device_I);
    free(device_J);
    free(device_V);
    free(device_x);
    free(device_b);
#endif

    return 0;
}
