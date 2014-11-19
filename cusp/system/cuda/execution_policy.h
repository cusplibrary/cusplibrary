/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

/*! \file thrust/system/cpp/execution_policy.h
 *  \brief Execution policies for Thrust's CUDA system.
 */

#include <thrust/detail/config.h>

// get the execution policies definitions first
#include <thrust/system/cuda/detail/execution_policy.h>

// get the definition of par
#include <thrust/system/cuda/detail/par.h>

namespace cusp
{
namespace system
{
namespace cuda
{
using namespace thrust::system::cuda;
} // end namespace cuda
} // end namespace system
} // end namespace cusp

// now get all the algorithm definitions

#include <cusp/system/cuda/detail/convert.h>
#include <cusp/system/cuda/detail/elementwise.h>
#include <cusp/system/cuda/detail/multiply.h>
#include <cusp/system/cuda/detail/transpose.h>
