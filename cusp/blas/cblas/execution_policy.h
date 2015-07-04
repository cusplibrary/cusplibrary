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
#include <cusp/blas/blas_policy.h>

namespace cusp
{
namespace blas
{
namespace cblas
{

struct execution_policy
        : public thrust::host_execution_policy<execution_policy>
{
    execution_policy(void) {}
};

} // end cblas
} // end blas

static const blas::cblas::execution_policy cblas;

} // end cusp

