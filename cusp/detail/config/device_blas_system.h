/*
 *  Copyright 2008-2012 NVIDIA Corporation
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

// reserve 0 for undefined
#define CUSP_DEVICE_BLAS_THRUST    1
#define CUSP_DEVICE_BLAS_CUBLAS    2
#define CUSP_DEVICE_BLAS_CBLAS     3

#ifndef CUSP_DEVICE_BLAS_SYSTEM
#define CUSP_DEVICE_BLAS_SYSTEM CUSP_DEVICE_BLAS_THRUST
#endif // CUSP_DEVICE_BLAS_BACKEND

#define CUSP_DEVICE_BACKEND_THRUST CUSP_DEVICE_BLAS_THRUST
#define CUSP_DEVICE_BACKEND_CUBLAS CUSP_DEVICE_BLAS_CUBLAS
#define CUSP_DEVICE_BACKEND_CBLAS  CUSP_DEVICE_BLAS_CBLAS

#if CUSP_DEVICE_BLAS_SYSTEM == CUSP_DEVICE_BLAS_THRUST
#define __CUSP_DEVICE_BLAS_NAMESPACE thrustblas
#elif CUSP_DEVICE_BLAS_SYSTEM == CUSP_DEVICE_BLAS_CUBLAS
#define __CUSP_DEVICE_BLAS_NAMESPACE cublas
#elif CUSP_DEVICE_BLAS_SYSTEM == CUSP_DEVICE_BLAS_CBLAS
#define __CUSP_DEVICE_BLAS_NAMESPACE cblas
#endif

#define __CUSP_DEVICE_BLAS_ROOT cusp/blas/__CUSP_DEVICE_BLAS_NAMESPACE

