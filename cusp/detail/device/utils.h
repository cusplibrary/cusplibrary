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

// ceil(x/y) for integers, used to determine # of blocks/warps etc.
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)

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


namespace cusp
{
namespace detail
{    
namespace device
{    

// Atomic operations are only available for capability 1.1 and above
#if __CUDA_ARCH__ >= 110

//
// We have to emulate FP atomicAdd because it isn't supported natively
// on devices prior to capability 2.0.
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
    } while( float_as_int(assumed)!=float_as_int(old) );
    //
    // NOTE: Comparing as ints rather than floats is mandatory.
    // For floats, the test NaN==NaN would always fall, leading to an
    // infinite loop.

    return old;
}

#if __CUDA_ARCH__ >= 130
//
// Double precision atomics are not supported on any device, so we
// always emulate with atomicCAS().
//
static __inline__ __device__ double atomicAdd(double *addr, double val)
{
    double old=*addr, assumed;
    
    do {
        assumed = old;
        old = __longlong_as_double( atomicCAS((unsigned long long int*)addr,
                                        __double_as_longlong(assumed),
                                        __double_as_longlong(val+assumed)));
    } while( __double_as_longlong(assumed)!=__double_as_longlong(old) );

    return old;
}
#endif // __CUDA_ARCH__ >= 130

#endif // __CUDA_ARCH__ >= 110

} // end namespace device
} // end namespace detail
} // end namespace cusp

