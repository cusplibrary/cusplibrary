#include <cusp/array2d.h>
#include <cusp/print.h>

#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#else
#include <cstdlib>
#include <cstring>
#endif

// This example shows how to wrap a 2-dimensional array in
// "raw" memory on the device.  The matrix has dimensions 2x3
// and is stored in row major format.  Note that each row of the
// matrix has one padding element so the "pitch" is 4.
//
//  Example Matrix (X is padding)
//   [ 1  2  3  X]
//   [ 4  5  6  X]


int main(void)
{
    // padding
    int X = -1;

    // 2d array in host memory
    int   host_A[8] = { 1, 2, 3, X, 4, 5, 6, X};

    // allocate memory for 2d array
    int * device_A;
#ifdef __CUDACC__
    cudaMalloc(&device_A, 8 * sizeof(int));
    cudaMemcpy(device_A, host_A, 8 * sizeof(int), cudaMemcpyHostToDevice);
    thrust::device_ptr<int> wrapped_device_A(device_A);
    typedef typename cusp::array1d_view< thrust::device_ptr<int> > Array1dView;
#else
    device_A = (int*)malloc(8 * sizeof(int));
    memcpy(device_A, host_A, 8 * sizeof(int));
    int* wrapped_device_A = device_A;
    typedef typename cusp::array1d_view<int*> Array1dView;
#endif

    // use array1d_view to represent the linear array data
    Array1dView linear_array(wrapped_device_A, wrapped_device_A + 8);

    // use array2d_view to wrap the linear array
    typedef cusp::array2d_view<Array1dView, cusp::row_major> Array2dView;

    // construct a array2d_view to the 2x3 array with pitch=4
    Array2dView A(2, 3, 4, linear_array);

    // print the wrapped arrays
    cusp::print(linear_array);
    cusp::print(A);

    // free arrays
#ifdef __CUDACC__
    cudaFree(device_A);
#else
    free(device_A);
#endif

    return 0;
}
