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

#include <cusp/detail/config.h>

#include <cusp/system/cuda/detail/multiply/coo_flat_spmv.h>
#include <cusp/system/cuda/detail/multiply/csr_vector_spmv.h>
#include <cusp/system/cuda/detail/multiply/dia_spmv.h>
#include <cusp/system/cuda/detail/multiply/ell_spmv.h>
// #include <cusp/system/cuda/detail/multiply/hyb_spmv.h>

#include <cusp/system/cuda/detail/multiply/spgemm.h>

namespace cusp
{
namespace system
{
namespace cuda
{

} // end namespace cuda
} // end namespace system

// hack until ADL is operational
using cusp::system::cuda::multiply;

} // end namespace cusp
