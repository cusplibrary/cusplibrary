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

// A simple timer class

#include <cuda.h>

class timer
{
    cudaEvent_t start;
    cudaEvent_t end;

    public:
    timer()
    { 
        CUDA_SAFE_CALL(cudaEventCreate(&start)); 
        CUDA_SAFE_CALL(cudaEventCreate(&end));
        CUDA_SAFE_CALL(cudaEventRecord(start,0));
    }

    ~timer()
    {
        CUDA_SAFE_CALL(cudaEventDestroy(start));
        CUDA_SAFE_CALL(cudaEventDestroy(end));
    }

    float milliseconds_elapsed()
    { 
        float elapsed_time;
        CUDA_SAFE_CALL(cudaEventRecord(end, 0));
        CUDA_SAFE_CALL(cudaEventSynchronize(end));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time, start, end));
        return elapsed_time;
    }
    float seconds_elapsed()
    { 
        return milliseconds_elapsed() / 1000.0;
    }
};


