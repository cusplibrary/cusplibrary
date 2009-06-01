/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */



#pragma once

#include <assert.h>
#include <cuda.h>
#include <cmath>


//macro to enforce intrawarp sychronization during emulation
#ifdef __DEVICE_EMULATION__
#define EMUSYNC __syncthreads()
#else
#define EMUSYNC
#endif

#  define CUDA_SAFE_CALL_NO_SYNC( call) do {                                 \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#  define CUDA_SAFE_CALL( call) do {                                         \
    CUDA_SAFE_CALL_NO_SYNC(call);                                            \
    cudaError err = cudaThreadSynchronize();                                 \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)



// ceil(x/y) for integers, used to determine # of blocks/warps etc.
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)

//#define small_grid_thread_id(void) ((blockDim.x * blockIdx.x + threadIdx.x))
//#define large_grid_thread_id(void) ((blockDim.x * (blockIdx.x + blockIdx.y*gridDim.x) + threadIdx.x))
#define small_grid_thread_id(void) ((__umul24(blockDim.x, blockIdx.x) + threadIdx.x))
#define large_grid_thread_id(void) ((__umul24(blockDim.x,blockIdx.x + __umul24(blockIdx.y,gridDim.x)) + threadIdx.x))


namespace cusp
{

namespace device
{    



#if !defined(CUDA_NO_SM_11_ATOMIC_INTRINSICS)
//
// We have to emulate FP atomicAdd because it isn't supported natively.
//
// Note that the semantics of atomicCAS() on float values is a little
// dodgy.  It just does a bit comparison rather than a true floating
// point comparison.  Hence 0 != -0, for instance.
//
static __inline__ __device__ float atomicAdd(float *addr, float val)
{
    float old=*addr, assumed;
    
    do {
        assumed = old;
        old = int_as_float( atomicCAS((int*)addr,
                                        float_as_int(assumed),
                                        float_as_int(val+assumed)));
    } while( assumed!=old );

    return old;
}
#endif // !defined(CUDA_NO_SM_11_ATOMIC_INTRINSICS)

#if !defined(CUDA_NO_SM_13_DOUBLE_INTRINSICS)
static __inline__ __device__ double atomicAdd(double *addr, double val)
{
    double old=*addr, assumed;
    
    do {
        assumed = old;
        old = __longlong_as_double( atomicCAS((unsigned long long int*)addr,
                                        __double_as_longlong(assumed),
                                        __double_as_longlong(val+assumed)));
    } while( assumed!=old );

    return old;
}
#endif // !defined(CUDA_NO_SM_13_DOUBLE_INTRINSICS)



/*
 *  For a given number of blocks, return a 2D grid large enough to contain them
 */
inline dim3 make_large_grid(const unsigned int num_blocks){
    if (num_blocks <= 65535){
        return dim3(num_blocks);
    } else {
        unsigned int side = (unsigned int) ceil(sqrt((double)num_blocks));
        return dim3(side,side);
    }
}

inline dim3 make_large_grid(const unsigned int num_threads, const unsigned int blocksize){
    const unsigned int num_blocks = DIVIDE_INTO(num_threads, blocksize);
    if (num_blocks <= 65535){
        //fits in a 1D grid
        return dim3(num_blocks);
    } else {
        //2D grid is required
        const unsigned int side = (unsigned int) ceil(sqrt((double)num_blocks));
        return dim3(side,side);
    }
}

inline dim3 make_small_grid(const unsigned int num_blocks){
    if (num_blocks <= 65535){
        return dim3(num_blocks);
    } else {
        fprintf(stderr,"Requested size exceedes 1D grid dimensions\n");
        return dim3(0);
    }
}

inline dim3 make_small_grid(const unsigned int num_threads, const unsigned int blocksize){
    const unsigned int num_blocks = DIVIDE_INTO(num_threads, blocksize);
    if (num_blocks <= 65535){
        return dim3(num_blocks);
    } else {
        fprintf(stderr,"Requested size exceedes 1D grid dimensions\n");
        return dim3(0);
    }
}


/*
 * Set the first N values of dest[] equal to val
 *
 *  Uses all avaible threads in a block.  Works for shared memory too.
 */
template <typename T>
__device__ void memset_device(T * dest, const T val, unsigned int num_values){

    unsigned int threads_per_block = blockDim.x*blockDim.y*blockDim.z;  
    unsigned int num_iters = num_values / threads_per_block;

    //set the bulk of the data
    //each thread sets one T at a time
    unsigned int index = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;   //threadId 
    
    for(unsigned int i = 0; i < num_iters; i++){
        dest[index] = val;              
        index += threads_per_block;
    }

    //set the tail of the data, some threads idle
    if(index < num_values){     
        dest[index] = val;
    }

    // Synchronize to make sure data is copied
    __syncthreads();
}





/*
 * Copy the first N values from src[] to dest[]
 */
template <typename IndexType, typename ValueType>
__device__ void memcpy_device(ValueType * dest, const ValueType * src, const IndexType num_values){ 
    for(unsigned int i = threadIdx.x; i < num_values; i += blockDim.x){
        dest[i] = src[i];       
    }
}



/*
 *  Gather from src into dest according to map.  Equivalent to the following C
 *
 *     for(i = 0; i < N; N++)
 *         dest[i] = src[map[i]]
 */
template <typename IndexType, typename ValueType>
__global__ void
gather_dev_kernel(ValueType * dest, const ValueType * src, const IndexType* map, const size_t N)
{   
    IndexType i = large_grid_thread_id();
    
    if(i < N){      
        dest[i] = src[map[i]];
    }
}

template <typename IndexType, typename ValueType>
void gather_device(ValueType * dest, const ValueType * src, const IndexType* map, const size_t N)
{
    const unsigned int ctasize = 256;
    const dim3 grid = make_large_grid(N, ctasize);

    gather_dev_kernel<IndexType,ValueType><<<grid, ctasize>>>(dest,src,map,N);
}

/*
 *  Scatter src into dest according to map.  Equivalent to the following C
 *
 *     for(i = 0; i < N; N++)
 *         dest[map[i]] = src[i]
 */
//template <typename IndexType, typename ValueType>
//__global__ void
//scatter_dev_kernel(ValueType * dest, const ValueType * src, const IndexType* map, const IndexType N)
//{ 
//    IndexType i = large_grid_thread_id();
//  
//  if(i < N){      
//      dest[map[i]] = src[i];
//  }
//}
//
//template <typename IndexType, typename ValueType>
//void scatter_device(ValueType * dest, const ValueType * src, const IndexType* map, const IndexType N)
//{
//    dim3 block(256);
//    dim3 grid = make_large_grid((unsigned int) ceil(N / (double) block.x));
//
//    scatter_dev_kernel<IndexType,ValueType><<<grid,block>>>(dest,src,map,N);
//}


} // end namespace device

} // end namespace cusp

