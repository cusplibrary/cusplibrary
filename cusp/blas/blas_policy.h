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

#include <cusp/detail/config.h>

#define __CUSP_HOST_BLAS_SYSTEM <__CUSP_HOST_BLAS_ROOT/blas.h>
#include __CUSP_HOST_BLAS_SYSTEM
#undef __CUSP_HOST_BLAS_SYSTEM

#define __CUSP_DEVICE_BLAS_SYSTEM <__CUSP_DEVICE_BLAS_ROOT/blas.h>
#include __CUSP_DEVICE_BLAS_SYSTEM
#undef __CUSP_DEVICE_BLAS_SYSTEM

#include <cusp/blas/thrustblas/blas.h>

